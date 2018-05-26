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

#include "factory.h" // network.h, optionsdict.h
#include "blas_transforms.h"
#include <cassert>
#include <cmath>

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

// Note that BlasNetworkComputation and BlasNetwork::evaluate are identical
// to their OpenCL counterpart, and the constructors are nearly so

namespace lczero {

class BlasNetwork;

class BlasNetworkComputation : public NetworkComputation {

 public:
  BlasNetworkComputation(BlasNetwork* network) : network_(network) {}

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

  void ComputeBlocking() override;

 private:
  std::vector<InputPlanes> inputs_;
  std::vector<float>   output_values_;
  std::vector<std::vector<float>> output_policies_;
  BlasNetwork* network_;
};

class BlasNetwork : public Network {
 public:
  BlasNetwork(const Weights& weights, const OptionsDict& options);

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<BlasNetworkComputation>(this);
  }

  // this is the main function used by outside code
  std::pair<float, std::vector<float>> evaluate(InputPlanes& input) const;

 private:
  // forwardPass is the actual network computation; evaluate() wraps this.
  void forwardPass(const std::vector<float>& input,
                         std::vector<float>& policy_data,
                         std::vector<float>& value_data) const;

  Weights weights_;
  const OptionsDict& options_;
};

void BlasNetworkComputation::ComputeBlocking() {
  //printf("evaluating batch of %lu nodes\n", inputs_.size());
  output_values_.resize(inputs_.size());
  output_policies_.resize(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); i++)
   std::tie(output_values_[i], output_policies_[i]) = network_->evaluate(inputs_[i]);
}

BlasNetwork::BlasNetwork(const Weights& weights, const OptionsDict& options)
  : weights_(weights), options_(options) {
  // this matches old Network::initialize, in Network.cpp:375-416
  BlasTransforms::initOneBlock(weights_.input, true);
  for (auto& resblock : weights_.residual) {
    BlasTransforms::initOneBlock(resblock.conv1);
    BlasTransforms::initOneBlock(resblock.conv2);
  }
  BlasTransforms::initOneBlock(weights_.policy, false, true);
  BlasTransforms::initOneBlock(weights_.value, false, true);
  printf("blas init complete\n");
}

std::pair<float, std::vector<float>> BlasNetwork::evaluate(InputPlanes& inputplanes) const {
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
  forwardPass(input_data, policy_data, value_data);
  //printf("Network forward pass complete, raw output:\n");
  //for (size_t i = 0; i < value_data.size(); i++)
  //  printf("%g ", value_data[i]);
  //printf("\n");
  //for (size_t i = 0; i < policy_data.size(); i++)
  //  printf("%g ", policy_data[i]);

  std::vector<float> output(weights_.ip2_val_b.size());
  BlasTransforms::innerproduct(value_data, weights_.ip2_val_w, weights_.ip2_val_b, output);
  assert(output.size() == 1);
  auto value = output[0];

  // normalize outputs
  auto policy = BlasTransforms::softmax(policy_data);
  value = std::tanh(value);
  //printf("returning network evaluation %g\n", value);
  return std::pair<float, std::vector<float>>(value, policy);
}

void BlasNetwork::forwardPass(const std::vector<float>& input,
                              std::vector<float>& output_pol,
                              std::vector<float>& output_val) const {
    // Input convolution
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int tiles = width * height / 4;
    // Calculate output channels
    const auto output_channels = weights_.input.biases.size();
    //input_channels is the maximum number of input channels of any convolution.
    //Residual blocks are identical, but the first convolution might be bigger
    //when the network has very few filters
    const auto input_channels = std::max(
            static_cast<size_t>(output_channels),
            static_cast<size_t>(kInputPlanes));
    auto conv_out = std::vector<float>(output_channels * width * height);

    auto V = std::vector<float>(WINOGRAD_TILE * input_channels * tiles);
    auto M = std::vector<float>(WINOGRAD_TILE * output_channels * tiles);

    std::vector<float> policy_data(weights_.policy.bn_means.size() * width * height); // NUM_POLICY_INPUT_PLANES*w*h
    std::vector<float> value_data(weights_.value.bn_means.size() * width * height); // NUM_VALUE_INPUT_PLANES*w*h

    BlasTransforms::winograd_convolve3(output_channels, input, weights_.input.weights, V, M, conv_out);
    BlasTransforms::batchnorm(output_channels, conv_out,
                              weights_.input.bn_means,
                              weights_.input.bn_stddivs);

    // Residual tower
    auto conv_in = std::vector<float>(output_channels * width * height);
    auto res = std::vector<float>(output_channels * width * height);
    for (auto& resblock : weights_.residual) {
      auto output_channels = resblock.conv1.biases.size(); // really confusing overload of variable names.... gcp pls...
      std::swap(conv_out, conv_in);
      std::copy(begin(conv_in), end(conv_in), begin(res));
      BlasTransforms::winograd_convolve3(output_channels, conv_in,
                                         resblock.conv1.weights, V, M, conv_out);
      BlasTransforms::batchnorm(output_channels, conv_out,
                                resblock.conv1.bn_means,
                                resblock.conv1.bn_stddivs);

      output_channels = resblock.conv2.biases.size();
      std::swap(conv_out, conv_in);
      BlasTransforms::winograd_convolve3(output_channels, conv_in,
                                         resblock.conv2.weights, V, M, conv_out);
      BlasTransforms::batchnorm(output_channels, conv_out,
                                resblock.conv2.bn_means,
                                resblock.conv2.bn_stddivs,
                                res.data());
    }
    BlasTransforms::convolve(weights_.policy.bn_means.size(), conv_out, // NUM_POLICY_INPUT_PLANES
                             weights_.policy.weights, weights_.policy.biases, policy_data);
    BlasTransforms::batchnorm(weights_.policy.bn_means.size(), policy_data, // NUM_POLICY_INPUT_PLANES
                              weights_.policy.bn_means, weights_.policy.bn_stddivs);
    BlasTransforms::innerproduct(policy_data, weights_.ip_pol_w, weights_.ip_pol_b, output_pol);

    BlasTransforms::convolve(weights_.value.bn_means.size(), conv_out, // NUM_VALUE_INPUT_PLANES
                             weights_.value.weights, weights_.value.biases, value_data);
    BlasTransforms::batchnorm(weights_.value.bn_means.size(), value_data, // NUM_VALUE_INPUT_PLANES
                              weights_.value.bn_means, weights_.value.bn_stddivs);
    BlasTransforms::innerproduct(value_data, weights_.ip1_val_w, weights_.ip1_val_b, output_val, true); // value head gets relu applied
}

REGISTER_NETWORK("blas", BlasNetwork, 80)

} // namespace lczero
