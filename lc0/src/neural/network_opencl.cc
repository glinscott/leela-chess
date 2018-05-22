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

#include "network_blas_cl_common.h" // factory.h, network.h, optionsdict.h
#include "opencl/OpenCLScheduler.h"

namespace lczero {

class OpenCLNetwork : public BlasCLNetwork {
 public:
  OpenCLNetwork(const Weights& weights, const OptionsDict& options);

 protected:
  void initialize(void);
  static std::vector<float> zeropad_U(const std::vector<float>& U, const int outputs, const int channels, const int outputs_pad, const int channels_pad);
  void forwardPass(const std::vector<float>& input_data,
                                std::vector<float>& policy_data,
                                std::vector<float>& value_data) override;
  OpenCLScheduler opencl_;
};

OpenCLNetwork::OpenCLNetwork(const Weights& weights, const OptionsDict& options)
  : BlasCLNetwork(weights, options) {
  // this function corresponds to Network.cpp:418-496
  //myprintf("Initializing OpenCL.\n");
  auto channels = weights_.input.biases.size();
  opencl_.initialize(channels);

  for (auto& opencl_net : opencl_.get_networks()) {
    auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();
    auto mwg = tuners[0];
    auto kwg = tuners[2];
    auto vwm = tuners[3];
    size_t m_ceil = ceilMultiple(ceilMultiple(channels, mwg), vwm);
    size_t k_ceil = ceilMultiple(ceilMultiple(kInputPlanes, kwg), vwm);

    auto Upad = zeropad_U(weights_.input.weights, channels, kInputPlanes, m_ceil, k_ceil);

    opencl_net->push_input_convolution(WINOGRAD_ALPHA, kInputPlanes, channels,
                                       Upad, weights_.input.bn_means, weights_.input.bn_stddivs);

    for (auto& resblock : weights_.residual) {
      auto Upad1 = zeropad_U(resblock.conv1.weights, channels, channels, m_ceil, m_ceil);
      auto Upad2 = zeropad_U(resblock.conv2.weights, channels, channels, m_ceil, m_ceil);
      opencl_net->push_residual(WINOGRAD_ALPHA, channels, channels,
                                Upad1, resblock.conv1.bn_means, resblock.conv1.bn_stddivs,
                                Upad2, resblock.conv2.bn_means, resblock.conv2.bn_stddivs);
    }

    constexpr unsigned int width = 8;
    constexpr unsigned int height = 8;
    const auto num_p_inputs  = weights_.policy.bn_means.size(); // NUM_POLICY_INPUT_PLANES
    const auto num_p_outputs = weights_.ip_pol_b.size();        // get_num_output_policy()
    const auto num_v_inputs  = weights_.value.bn_means.size();  // NUM_VALUE_INPUT_PLANES
    const auto num_v_outputs = weights_.ip1_val_b.size();       // NUM_VALUE_CHANNELS

    opencl_net->push_policy(channels, num_p_inputs, num_p_inputs*width*height, num_p_outputs,
                            weights_.policy.weights,
                            weights_.policy.bn_means, weights_.policy.bn_stddivs,
                            weights_.ip_pol_w, weights_.ip_pol_b);

    opencl_net->push_value (channels, num_v_inputs, num_v_inputs*width*height, num_v_outputs,
                            weights_.value.weights,
                            weights_.value.bn_means, weights_.value.bn_stddivs,
                            weights_.ip1_val_w, weights_.ip1_val_b);

  }
}

std::vector<float> OpenCLNetwork::zeropad_U(const std::vector<float>& U, const int outputs, const int channels, const int outputs_pad, const int channels_pad) {
  // Fill with zeroes
  auto Upad = std::vector<float>(WINOGRAD_TILE * outputs_pad * channels_pad);

  for(auto o = 0; o < outputs; o++) {
      for(auto c = 0; c < channels; c++) {
          for(auto xi = 0; xi < WINOGRAD_ALPHA; xi++){
              for(auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                  Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                     + nu * (outputs_pad * channels_pad)
                     + c * outputs_pad +
                       o]
                  =
                  U  [xi * (WINOGRAD_ALPHA * outputs * channels)
                    + nu * (outputs * channels)
                    + c * outputs
                    + o];
              }
          }
      }
  }

  return Upad;
}

inline void OpenCLNetwork::forwardPass(const std::vector<float>& input_data,
                              std::vector<float>& policy_data,
                              std::vector<float>& value_data) {
  opencl_.forward(input_data, policy_data, value_data);
}

REGISTER_NETWORK("opencl", OpenCLNetwork, 100)

} // namespace lczero
