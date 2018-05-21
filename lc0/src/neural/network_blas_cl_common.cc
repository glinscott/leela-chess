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

namespace lczero {

void BlasCLNetworkComputation::ComputeBlocking() override {
  output_values = std::make_unique<float[]>(inputs.size());
  output_policies = std::make_unique<float[][]>(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++)
    output_values[i], output_policies[i] = network->evaluate(inputs[i]);
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

void BlasCLNetwork::initOneBlock(Weights::ConvBlock& block, bool inputlayer=false) {

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

std::vector<float> BlasCLNetwork::softmax(const std::vector<float>& input, float temperature) {
  std::vector<float>  output(input.size());
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

std::pair<float value, float[] policy> evaluate(InputPlanes&& input) {
  // something something convert input data
  forward(input_data, policy_data, value_data); // virtual
  
  // Get the moves
  softmax(policy_data, softmax_data, cfg_softmax_temp);
  std::vector<float>& outputs = softmax_data;

  // Now get the score
  innerproduct<NUM_VALUE_CHANNELS, 1>(value_data, ip2_val_w, ip2_val_b, winrate_out);
  
  // blah blah further processing
  
  float[] policy = std::make_unique<float[]>();
  
  // blah blah
  
  return {value, policy};
}

} // namespace lczero
