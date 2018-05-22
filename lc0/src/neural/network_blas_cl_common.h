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

/* A table of variable conversions from lczero/Network.cpp to lc0/network.h

get_input_channels()     -> kInputPlanes
Network::initialize::channels -> ConvBlock.biases.size()
conv_weights             -> ConvBlock.weights
conv_biases              -> ConvBlock.biases
batchnorm_means          -> ConvBlock.bn_means
batchnorm_stddivs        -> ConvBlock.bn_stddivs
conv_pol_w               -> Weights.policy.weights
conv_pol_b               -> Weights.policy.biases
bn_pol_w1                -> Weights.policy.bn_means,
bn_pol_w2                -> Weights.policy.bn_stddivs // ??? not very useful oldnames
NUM_POLICY_INPUT_PLANES  -> Weights.policy.bn_means.size()
ip_pol_w                 -> Weights.ip_pol_w
ip_pol_b                 -> Weights.ip_pol_b
get_num_output_policy()  -> Weights.ip_pol_b.size()
conv_val_w               -> Weights.value.weights
conv_val_b               -> Weights.value.biases
bn_val_w1                -> Weights.value.bn_means
bn_val_w2                -> Weights.value.bn_stddivs
NUM_VALUE_INPUT_PLANES   -> Weights.value.bn_means.size()
{ip1_val_w,ip1_val_b,ip2_val_w,ip2_val_b} -> Weights.{}
NUM_VALUE_CHANNELS       -> Weights.ip1_val_b.size()

*/

#pragma once

#include "factory.h" // network.h, optionsdict.h
#include "opencl/blas_config.h"

namespace lczero {

class BlasCLNetwork;

class BlasCLNetworkComputation : public NetworkComputation {

 public:
  BlasCLNetworkComputation(BlasCLNetwork* network) : network(network) {}

  void AddInput(InputPlanes&& input) override {
    inputs.push_back(input);
  }

  int GetBatchSize() const override {
    return inputs.size();
  }

  float GetQVal(int sample) const override {
    return output_values[sample];
  }

  float GetPVal(int sample, int move_id) const override {
    return output_policies[sample][move_id];
  }

  void ComputeBlocking() override;

 private:
  std::vector<InputPlanes> inputs;
  std::vector<float>   output_values;
  std::vector<std::vector<float>> output_policies;
  BlasCLNetwork* network;
};

class BlasCLNetwork : public Network {
 public:
  BlasCLNetwork(const Weights& weights, const OptionsDict& options)
    : weights(weights), options(options) {
    initialize();
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasCLNetworkComputation>(this);
  }

  std::pair<float, std::vector<float>> evaluate(InputPlanes& input);

 protected:
  virtual void forwardPass(const std::vector<float>& input,
                                 std::vector<float>& policy_data,
                                 std::vector<float>& value_data);

  std::vector<float> softmax(const std::vector<float>& input, float temperature=1.0f);
  // TODO: softmaxtemp hardcoded from lczero

  std::vector<float> innerproduct(const std::vector<float>& inputs,
                                  const std::vector<float>& weights,
                                  const std::vector<float>& biases,
                                  bool apply_relu=false);

  void initialize(void);
  void initOneBlock(Weights::ConvBlock& block, bool inputlayer=false);
  std::vector<float> winograd_transform_f(const std::vector<float>& f, const int outputs, const int channels);

  Weights weights; // optimal memory use? is one reference shared among multiple backends?
  const OptionsDict& options;
};

} // namespace lczero