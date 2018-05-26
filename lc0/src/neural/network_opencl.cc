/*
    This file is part of Leela Chess Zero.
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

#include "factory.h" // network.h, optionsdict.h
#include "blas_transforms.h"
#include "opencl/OpenCLParams.h"
#include "opencl/OpenCL.h"

// Note that OpenCLNetworkComputation and OpenCLNetwork::evaluate are identical
// to their Blas counterpart, and the constructors are nearly so

namespace lczero {

// literally copy and pasted from network_blas.cc
class OpenCLNetworkComputation : public NetworkComputation {

 public:
  OpenCLNetworkComputation(OpenCLNetwork* network) : network_(network) {}

  void AddInput(InputPlanes&& input) override {
    inputs_.emplace_back(input);
  }

  int GetBatchSize() const override {
    return inputs_.size();
  }

  float GetQVal(int sample) const override {
    return output_values_[sample];
  }

  float GetPVal(int sample, int move_id) const override {
    return output_policies_[sample][move_id];
  }

  void ComputeBlocking() override {
    //printf("evaluating batch of %lu nodes\n", inputs_.size());
    output_values_.resize(inputs_.size());
    output_policies_.resize(inputs_.size());
    for (size_t i = 0; i < inputs_.size(); i++)
      std::tie(output_values_[i], output_policies_[i]) = network_->evaluate(inputs_[i]);
};

 private:
  std::vector<InputPlanes> inputs_;
  std::vector<float>   output_values_;
  std::vector<std::vector<float>> output_policies_;
  OpenCLNetwork* network_;
};

class OpenCLNetwork : public Network {

 public:
  OpenCLNetwork(const Weights& weights, const OptionsDict& options);

  std::pair<float, std::vector<float>> evaluate(InputPlanes& input) /*const*/;

  OpenCLParams params_;
  OpenCL opencl_;
  OpenCL_Network opencl_net_;
  Weights::Vec ip2_val_w_; // final value head matmul is on cpu
  Weights::Vec ip2_val_b_;
};

using BlasTransforms;

OpenCLNetwork::OpenCLNetwork(const Weights& weights, const OptionsDict& options)
    params_(), opencl_(), opencl_net_(opencl_),
    ip2_val_w_(weights.ip2_val_w), ip2_val_b_(weights.ip2_val_b) {

  Weights weights_ = weights; // scratch weights, to be discarded

  params_.gpuId=options.GetOrDefault<int>("gpu", -1);
  params_.verbose=options.GetOrDefault<bool>("verbose", false);
  params_.force_tune=options.GetOrDefault<int>("force_tune", false);
  params_.tune_only=options.GetOrDefault<int>("tune_only", false);
  params_.tune_exhaustive=options.GetOrDefault<int>("tune_exhaustive", false);

  // this function corresponds to Network.cpp:418-496
  printf("Initializing OpenCL.\n");
  auto channels = weights_.input.biases.size();
  opencl_.initialize(channels, params_);

  {
    auto tuners = opencl_net_.getOpenCL().get_sgemm_tuners();
    auto mwg = tuners[0];
    auto kwg = tuners[2];
    auto vwm = tuners[3];
    size_t m_ceil = ceilMultiple(ceilMultiple(channels, mwg), vwm);
    size_t k_ceil = ceilMultiple(ceilMultiple(kInputPlanes, kwg), vwm);

    initOneBlock(weights_.input, true);
    auto Upad = zeropad_U(weights_.input.weights, channels, kInputPlanes, m_ceil, k_ceil);
    opencl_net_.push_input_convolution(WINOGRAD_ALPHA, kInputPlanes, channels,
                                       Upad, weights_.input.bn_means, weights_.input.bn_stddivs);

    for (auto& resblock : weights_.residual) {
      initOneBlock(resblock.conv1);
      initOneBlock(resblock.conv2);
      auto Upad1 = zeropad_U(resblock.conv1.weights, channels, channels, m_ceil, m_ceil);
      auto Upad2 = zeropad_U(resblock.conv2.weights, channels, channels, m_ceil, m_ceil);
      opencl_net_.push_residual(WINOGRAD_ALPHA, channels, channels,
                                Upad1, resblock.conv1.bn_means, resblock.conv1.bn_stddivs,
                                Upad2, resblock.conv2.bn_means, resblock.conv2.bn_stddivs);
    }

    initOneBlock(weights_.policy, false, true);
    initOneBlock(weights_.value, false, true);
    constexpr unsigned int width = 8;
    constexpr unsigned int height = 8;
    const auto num_p_inputs  = weights_.policy.bn_means.size(); // NUM_POLICY_INPUT_PLANES
    const auto num_p_outputs = weights_.ip_pol_b.size();        // get_num_output_policy()
    const auto num_v_inputs  = weights_.value.bn_means.size();  // NUM_VALUE_INPUT_PLANES
    const auto num_v_outputs = weights_.ip1_val_b.size();       // NUM_VALUE_CHANNELS

    opencl_net_.push_policy(channels, num_p_inputs, num_p_inputs*width*height, num_p_outputs,
                            weights_.policy.weights,
                            weights_.policy.bn_means, weights_.policy.bn_stddivs,
                            weights_.ip_pol_w, weights_.ip_pol_b);

    opencl_net_.push_value (channels, num_v_inputs, num_v_inputs*width*height, num_v_outputs,
                            weights_.value.weights,
                            weights_.value.bn_means, weights_.value.bn_stddivs,
                            weights_.ip1_val_w, weights_.ip1_val_b);

  }
  // weights_ should be deleted now
  printf("OpenCL init complete\n");
}

// literally Ctrl+C -> Ctrl+V from network_blas.cc
std::pair<float, std::vector<float>> BlasNetwork::evaluate(InputPlanes& inputplanes) /*const*/ {
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
  auto value_data = std::vector<float>(weights_.ip1_val_b.size()); // NUM_VALUE_CHANNELS

  //printf("Network::evaluate: input parsed, calling network...\n");
  opencl_net_.forward(input_data, policy_data, value_data);
  //printf("Network forward pass complete, raw output:\n");
  //for (size_t i = 0; i < value_data.size(); i++)
  //  printf("%g ", value_data[i]);
  //printf("\n");
  //for (size_t i = 0; i < policy_data.size(); i++)
  //  printf("%g ", policy_data[i]);

  std::vector<float> output(ip2_val_b_.size());
  innerproduct(value_data, ip2_val_w_, ip2_val_b_, output);
  assert(output.size() == 1);
  auto value = output[0];

  // normalize outputs
  auto policy = softmax(policy_data);
  value = std::tanh(value);
  //printf("returning network evaluation %g\n", value);
  return std::pair<float, std::vector<float>>(value, policy);
}

REGISTER_NETWORK("opencl", OpenCLNetwork, 100)

} // namespace lczero
