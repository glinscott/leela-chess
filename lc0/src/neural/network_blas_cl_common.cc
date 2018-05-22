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

#include "network_blas_cl_common.h"
#include "utils/bititer.h"
#include <cassert>

namespace lczero {

void BlasCLNetworkComputation::ComputeBlocking() {
  output_values = std::vector<float>(inputs.size());
  output_policies = std::vector<std::vector<float>>(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++)
    std::tie(output_values[i], output_policies[i]) = network->evaluate(inputs[i]);
}
  

void BlasCLNetwork::initialize(void) {
  // this matches old Network::initialize, in Network.cpp:375-416   
  initOneBlock(weights.input, true);
  for (auto& resblock : weights.residual) {
    initOneBlock(resblock.conv1);
    initOneBlock(resblock.conv2);
  }
  initOneBlock(weights.policy);
  initOneBlock(weights.value);
}

void BlasCLNetwork::initOneBlock(Weights::ConvBlock& block, bool inputlayer) {

  size_t channels;
  if (inputlayer)
    channels = kInputPlanes;
  else
    channels = block.biases.size();
  block.weights = winograd_transform_f(block.weights, block.biases.size(), channels);

  // Biases are not calculated and are typically zero but some networks might
  // still have non-zero biases.
  // Move biases to batchnorm means to make the output match without having
  // to separately add the biases.
  for (auto j = size_t{0}; j < block.bn_means.size(); j++) {
    block.bn_means[j] -= block.biases[j];
    block.biases[j] = 0.0f;
  }
}

std::pair<float, std::vector<float>> BlasCLNetwork::evaluate(InputPlanes& inputplanes) {
  // thanks to Francois and crem for verifying
  auto input_data = std::vector<float>(kInputPlanes*64, 0.0); // get_input_channels()*w*h
  // the loop below assumes that input_data[i] is initialized to 0 ^
  size_t index = 0;
  for (auto& plane : inputplanes) {
    for (auto i : IterateBits(plane.mask)) {
      input_data[index+i] = plane.value;
    }
    index += 64;
  }
  assert(index == input_data.size());

  auto policy_data = std::vector<float>(weights.ip_pol_b.size()); // get_num_output_policy()
  auto  value_data = std::vector<float>(weights.value.bn_means.size()*64); //NUM_VALUE_INPUT_PLANES*64
  
  forwardPass(input_data, policy_data, value_data); // virtual
  
  // Get the moves
  auto policy = softmax(policy_data);

  // Now get the score
  auto output = innerproduct<weights.ip1_val_b.size(), 1>(value_data, weights.ip2_val_w, weights.ip2_val_b);
  assert(output.size() == 1);
  value = output[0];
  
  value = std::tanh(value);

  return {value, policy};
}

std::vector<float> BlasCLNetwork::winograd_transform_f(const std::vector<float>& f, const int outputs, const int channels) {
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

std::array<float> BlasCLNetwork::softmax(const std::vector<float>& input, float temperature) {
  auto output = std::make_unique<float[]>(input.size());
  auto alpha = *std::max_element(begin(input), begin(input) + output.size());
  alpha /= temperature
  auto demon = 0.0f;
  for (size_t i = 0; i < input.size(); i++) {
    auto val = std::exp((input[i]/temperature) - alpha);
    output[i] = val;
    denom += val;
  }
  for (size_t i = 0; i < input.size(); i++)
    output[i] /= denom;
  return out;
}

template<unsigned int inputs,
         unsigned int outputs,
         size_t W, size_t B>
         // TODO: surely this can be simplified? replace template array sizes with std::make_unique, or even replace arrays with vectors
std::vector<float> innerproduct(const std::vector<float>& input,
                   const std::array<float, W>& weights,
                   const std::array<float, B>& biases) {

  assert(B == outputs);
  auto output = std::vector<float>(outputs);

  cblas_sgemv(CblasRowMajor, CblasNoTrans,
              // M     K
              outputs, inputs,
              1.0f, &weights[0], inputs,
              &input[0], 1,
              0.0f, &output[0], 1);

  auto lambda_ReLU = [](float val) { return (val > 0.0f) ? val : 0.0f; };

  for (size_t o = 0; o < outputs; o++) {
    float val = biases[o] + output[o];
    if (outputs == weights.ip1_val_b.size()) { // NUM_VALUE_CHANNELS
      val = lambda_ReLU(val);
    }
    output[o] = val;
  }
  return output;
}

} // namespace lczero
