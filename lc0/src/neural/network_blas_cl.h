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

/* This file defines the BlasNetwork and OpenCLNetwork classes.

LeelaZero's original OpenCL backend was written using a few pieces of the blas
backend (including initialization) as helper code; meanwhile, the OpenCL
selfcheck also *calls* the CPU implementation to verify the GPU computations,
which means that it's probably more efficient if the CPU class and GPU class
share network weights, so the OpenCLNetwork just directly inherits from BlasNetwork.
Protected members are those which the OpenCL implementation uses, while private
members are those used exclusively by the CPU implementation.
*/

#pragma once

#include "factory.h" // network.h, optionsdict.h
#include "blas_config.h"
#include "opencl/OpenCLScheduler.h"

namespace lczero {

class BlasNetwork;

class BlasNetworkComputation : public NetworkComputation {

 public:
  BlasNetworkComputation(BlasNetwork* network) : network(network) {}

  void AddInput(InputPlanes&& input) override {
    inputs.emplace_back(input);
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
  BlasNetwork* network;
};

class BlasNetwork : public Network {

 public:
  BlasNetwork(const Weights& weights, const OptionsDict& options);

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasNetworkComputation>(this);
  }

  std::pair<float, std::vector<float>> evaluate(InputPlanes& input);
  // this is the main function used by outside code

 protected: // shared functions between blas and OpenCL implementations
  virtual void forwardPass(const std::vector<float>& input,
                                 std::vector<float>& policy_data,
                                 std::vector<float>& value_data);
  // forwardPass is the actual network computation; evaluate() wraps this.
  static std::vector<float> innerproduct(const std::vector<float>& inputs,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& biases,
                                         bool apply_relu=false);
  /* TODO: understand why gcp left the final value head computation out of OpenCL,
  leaving it to be done on the cpu. This is why forwardPass() doesn't simply return
  the evaluation, rather than the value_data array which needs to be processed by
  innerproduct(). If this last layer were done in OpenCL, then innerproduct() would
  not need to be shared. Best guess so far: for selfcheck purposes, it's easier
  to debug/compare the value head's second-to-last output vector, rather than
  its last output scalar?
  */

  void initOneBlock(Weights::ConvBlock& block, bool inputlayer=false);
  static std::vector<float> winograd_transform_f(const std::vector<float>& f, const int outputs, const int channels);
  static std::vector<float> softmax(const std::vector<float>& input, float temperature=1.0f);
  // TODO: softmaxtemp hardcoded from lczero

  Weights weights_; // optimal memory use? is one reference shared among multiple backends?
  const OptionsDict& options_;

 private: // functions specific to cpu-blas implementation
  void winograd_transform_in();
  void winograd_sgemm();
  void winograd_transform_out();
  void winograd_convolve3();
  template<unsigned int filter_size> void convolve();
  template<size_t spatial_size> void batchnorm();
};


class OpenCLNetwork : public BlasNetwork {

 public:
  OpenCLNetwork(const Weights& weights, const OptionsDict& options);

 private:
  void forwardPass(const std::vector<float>& input_data,
                                std::vector<float>& policy_data,
                                std::vector<float>& value_data) override;

  static std::vector<float> zeropad_U(const std::vector<float>& U,
                                      const int outputs, const int channels,
                                      const int outputs_pad,
                                      const int channels_pad);

  bool compare_net_outputs(const std::vector<float>& data,
                           const std::vector<float>& ref,
                           bool& fatal,
                           bool display_only = false,
                           std::string info = "");

  void doSelfCheck(const std::vector<float>& input_data,
                   const std::vector<float>& policy_data,
                   const std::vector<float>& value_data);

  static constexpr int SELFCHECK_PROBABILITY = 2000; // 1/2000
  static constexpr int SELFCHECK_MIN_EXPANSIONS = 2'000'000;

  OpenCLScheduler opencl_;
};

} // namespace lczero
