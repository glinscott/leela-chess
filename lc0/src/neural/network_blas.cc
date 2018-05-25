/*
    This file is part of Leela Chess Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto
    Copyright (C) 2018 The LCZero Authors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "network_blas_cl.h"
#include "utils/bititer.h"
#include <cblas.h>
#include <cassert>
#include <cmath>
#include <algorithm>

namespace lczero {

void BlasNetworkComputation::ComputeBlocking() {
  printf("evaluating batch of %d nodes\n", inputs.size());
  output_values = std::vector<float>(inputs.size());
  output_policies = std::vector<std::vector<float>>(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++)
    std::tie(output_values[i], output_policies[i]) = network->evaluate(inputs[i]);
  for (size_t i = 0; i < output_values.size(); i++)
    printf("node %d: evaluation %g\n", i, output_values[i]);
}

BlasNetwork::BlasNetwork(const Weights& weights, const OptionsDict& options)
  : weights_(weights), options_(options) {
  // this matches old Network::initialize, in Network.cpp:375-416
  initOneBlock(weights_.input, true);
  for (auto& resblock : weights_.residual) {
    initOneBlock(resblock.conv1);
    initOneBlock(resblock.conv2);
  }
  initOneBlock(weights_.policy, false, true);
  initOneBlock(weights_.value, false, true);
  printf("blas init complete\n");
}

void BlasNetwork::initOneBlock(Weights::ConvBlock& block, bool inputlayer, bool headlayer) {

  if (!headlayer) {
    size_t channels;
    if (inputlayer)
      channels = kInputPlanes;
    else
      channels = block.biases.size();
    block.weights = winograd_transform_f(block.weights, block.biases.size(), channels);
  }

  // the weights stored are actually variances, not stddivs, and also all the
  // code downstream assumes that they're inverse stddivs for efficiency
  constexpr float epsilon = 1e-5f;
  for (auto& weight : block.bn_stddivs) {
    weight = 1.0f / std::sqrt(weight + epsilon);
  }

  // Biases are not calculated and are typically zero but some networks might
  // still have non-zero biases.
  // Move biases to batchnorm means to make the output match without having
  // to separately add the biases.
  for (auto j = size_t{0}; j < block.bn_means.size(); j++) {
    block.bn_means[j] -= block.biases[j];
    block.biases[j] = 0.0f;
  }
}

void BlasNetwork::forwardPass(const std::vector<float>& input_data,
                              std::vector<float>& policy_data,
                              std::vector<float>& value_data) {

}

std::pair<float, std::vector<float>> BlasNetwork::evaluate(InputPlanes& inputplanes) {
  auto input_data = std::vector<float>(kInputPlanes*64, 0.0); // get_input_channels()*w*h
  size_t index = 0;
  for (auto& plane : inputplanes) {
    uint64_t shift = 1;
    for (int i = 0; i < 64; i++) {
      input_data[index++] = (plane.mask & shift) ? plane.value : 0;
      shift <<= 1;
    }
  }
  assert(index == input_data.size());

  auto policy_data = std::vector<float>(weights_.ip_pol_b.size()); // get_num_output_policy()
  auto  value_data = std::vector<float>(weights_.ip1_val_b.size()); //NUM_VALUE_CHANNELS

  printf("Network::evaluate: input parsed, calling network...\n");
  forwardPass(input_data, policy_data, value_data);
  printf("Network forward pass complete, raw output:\n");
  for (size_t i = 0; i < value_data.size(); i++)
    printf("%g ", value_data[i]);
  printf("\n");
  for (size_t i = 0; i < policy_data.size(); i++)
    printf("%g ", policy_data[i]);


  // Get the moves
  auto policy = softmax(policy_data);

  // Now get the score
  auto output = innerproduct(value_data, weights_.ip2_val_w, weights_.ip2_val_b);
  assert(output.size() == 1);
  auto value = output[0];

  value = std::tanh(value);
  printf("returning network evaluation\n");
  return std::pair<float, std::vector<float>>(value, policy);
}

std::vector<float> BlasNetwork::winograd_transform_f(const std::vector<float>& f, const int outputs, const int channels) {
  // F(2x2, 3x3) Winograd filter transformation
  // transpose(G.dot(f).dot(G.transpose()))
  // U matrix is transposed for better memory layout in SGEMM
  auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
  auto G = std::array<float, WINOGRAD_TILE>{1.0,  0.0,  0.0,
                                            0.5,  0.5,  0.5,
                                            0.5, -0.5,  0.5,
                                            0.0,  0.0,  1.0};
  auto temp = std::array<float, 12>{};

  for (auto o = 0; o < outputs; o++) {
    for (auto c = 0; c < channels; c++) {
      for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 3; j++) {
          auto acc = 0.0f;
          for (auto k = 0; k < 3; k++) {
            acc += G[i*3 + k] * f[o*channels*9 + c*9 + k*3 + j];
          }
          temp[i*3 + j] = acc;
        }
      }

      for (auto xi = 0; xi < 4; xi++) {
        for (auto nu = 0; nu < 4; nu++) {
          auto acc = 0.0f;
          for (int k = 0; k < 3; k++) {
            acc += temp[xi*3 + k] * G[nu*3 + k];
          }
          U[xi * (4 * outputs * channels)
            + nu * (outputs * channels)
            + c * outputs
            + o] = acc;
        }
      }
    }
  }

    return U;
}

std::vector<float> BlasNetwork::softmax(const std::vector<float>& inputs, float temperature) {
  auto outputs = std::vector<float>(inputs.size());
  auto alpha = *std::max_element(begin(inputs), begin(inputs) + outputs.size());
  alpha /= temperature;
  auto denom = 0.0f;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto val = std::exp((inputs[i]/temperature) - alpha);
    outputs[i] = val;
    denom += val;
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i] /= denom;
  }
  return outputs;
}

std::vector<float> BlasNetwork::innerproduct(const std::vector<float>& inputs,
                                               const std::vector<float>& weights,
                                               const std::vector<float>& biases,
                                               bool apply_relu) {
  auto outputs = std::vector<float>(biases.size());

  cblas_sgemv(CblasRowMajor, CblasNoTrans,
              // M           K
              biases.size(), inputs.size(),
              1.0f, weights.data(), inputs.size(),
              inputs.data(), 1, 0.0f,
              outputs.data(), 1);

  auto lambda_ReLU = [](float val) { return (val > 0.0f) ? val : 0.0f; };

  for (size_t o = 0; o < outputs.size(); o++) {
    float val = biases[o] + outputs[o];
    if (apply_relu) { // TODO: for value head fully connected layer in blas-cpu
      val = lambda_ReLU(val);
    }
    outputs[o] = val;
  }
  return outputs;
}

REGISTER_NETWORK("blas", BlasNetwork, 80)

} // namespace lczero