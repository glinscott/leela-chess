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

#pragma once

#include "factory.h" // network.h, optionsdict.h

namespace lczero {

class BlasCLNetworkComputation : public NetworkComputation {

  BlasCLNetworkComputation(const BlasCLNetwork* network) : network(network) {}

  void AddInput(InputPlanes&& input) override {
    inputs.push_back(input);
  }

  void GetBatchSize() override {
    return inputs.size();
  }

  void ComputeBlocking() override {
    output_values = std::make_unique<float[]>(inputs.size());
    output_policies = std::make_unique<float[][]>(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
      output_values[i], output_policies[i] = network->forward(inputs[i]);
  }
  
  int GetQVal(int sample) override {
    return output_values[sample];
  }
  
  float GetPVal(int sample, int move_id) override {
    return output_policies[sample][move_id];
  }

 private:
  std::vector<InputPlanes> inputs;
  float[]   output_values;
  float[][] output_policies;
  BlasNetwork* network;
}

class BlasCLNetwork : public Network {
 public:
  BlasCLNetwork(const Weights& weights, const OptionsDict& options)
    : weights(weights), options(options) {
    // this matches old Network::initialize, in Network.cpp:375-416   
    initOneBlock(weights.input, true);
    for (auto& resblock : weights.residual)
      initOneBlock(resblock);
    initOneBlock(weights.policy);
    initOneBlock(weights.value);
  }
    
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasCLNetworkComputation>(this);
  }
  virtual std::pair<float value, float[] policy> forwardEval(InputPlanes&& input);
  // output array should be std::make_unique
  
 protected:
  virtual void initOneBlock(Weights::ConvBlock& block, bool inputlayer=false);
  Weights::Vec winograd_transform_f(const Weights::Vec& f, const int outputs, const int channels);
  void softmax();

  Weights weights; // optimal memory use? is one reference shared among multiple backends?
  OptionsDict& options;
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

Weights::Vec BlasCLNetwork::winograd_transform_f(const Weights::Vec& f, const int outputs, const int channels) {

}

void BlasCLNetwork::softmax();
