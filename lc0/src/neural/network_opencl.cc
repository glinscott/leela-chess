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

#include "network_blas_cl.h" // factory.h, network.h, optionsdict.h
#include "utils/random.h"
#include <cstdio>

namespace lczero {

OpenCLNetwork::OpenCLNetwork(const Weights& weights, const OptionsDict& options)
  : BlasNetwork(weights, options) {
  // ^ nontrivial: all cpu initialization is shared by OpenCL initialization
  // this function corresponds to Network.cpp:418-496
  printf("Initializing OpenCL.\n");
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
  printf("OpenCL init complete\n");
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
  //printf("evaluating network...\n");
  opencl_.forward(input_data, policy_data, value_data);

#ifdef USE_OPENCL_SELFCHECK
  //if (Random::Get().GetInt(1, SELFCHECK_PROBABILITY) == 1) {
  //  doSelfCheck(input_data, policy_data, value_data);
  //}
#endif
}

// silly helper function
template<typename T>
T relative_difference(T a, T b) {
  // Handle NaN
  if (std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<T>::max();
  }

  constexpr auto small_number = 1e-3f;
  auto fa = std::fabs(a);
  auto fb = std::fabs(b);

  if (fa > small_number && fb > small_number) {
    // Handle sign difference
    if (((a < 0) != (b < 0)) && (a != 0) && (b != 0)) {
      return std::numeric_limits<T>::max();
    }
  }

  // Handle underflow
  fa = std::max(fa, small_number);
  fb = std::max(fb, small_number);

  return std::max(fabs((fa - fb) / fa), fabs((fa - fb) / fb));
}

bool OpenCLNetwork::compare_net_outputs(const std::vector<float>& data,
                                        const std::vector<float>& ref,
                                        bool& fatal,
                                        bool display_only,
                                        std::string info) const {
    bool almost_equal = true;
    // The idea is to allow an OpenCL error > 10% every SELFCHECK_MIN_EXPANSIONS
    // correct expansions. As the num_expansions increases between errors > 10%,
    // we'll allow more errors to occur (max 3) before crashing. As if it
    // builds up credit.
    constexpr unsigned int min_correct_expansions = SELFCHECK_MIN_EXPANSIONS / SELFCHECK_PROBABILITY / 2;
    static_assert(min_correct_expansions > 0, "Increase minimal nof expansions");
    static std::atomic<unsigned int> num_expansions{min_correct_expansions};
    num_expansions = std::min(num_expansions + 1, 3 * min_correct_expansions);

    // We accept an error up to 10%, but output values
    // smaller than 1/1000th are "rounded up" for the comparison.
    constexpr float relative_error = 0.1f;
    for (auto idx = size_t{0}; idx < data.size(); ++idx) {
        auto err = relative_difference(data[idx], ref[idx]);
        if (display_only) {
            printf("compare_net_outputs %s idx %d data %f ref %f err=%f\n",
                info.c_str(), idx, data[idx], ref[idx], err);
        } else if (err > relative_error) {
            almost_equal = false;
            printf("Error in OpenCL calculation: expected %f got %f (%lli"
                       "(error=%f%%)\n", ref[idx], data[idx], num_expansions.load(), err * 100.0);
            if (num_expansions < min_correct_expansions) {
                fatal = true;
            }
            else {
                num_expansions -= min_correct_expansions;
            }
        }
    }
    return almost_equal;
}

void OpenCLNetwork::doSelfCheck(const std::vector<float>& input_data,
                                const std::vector<float>& policy_data,
                                const std::vector<float>& value_data) /*const*/ {
  std::vector<float> cpu_policy_data(policy_data.size());
  std::vector<float> cpu_value_data(value_data.size());
  bool fatal = false;
  BlasNetwork::forwardPass(input_data, cpu_policy_data, cpu_value_data);
  bool almost_equal = compare_net_outputs(policy_data, cpu_policy_data, fatal);
  almost_equal &= compare_net_outputs(value_data, cpu_value_data, fatal);
  if (!almost_equal) {
    //printf("PGN\n%s\nEND\n", pos.pgn().c_str());
    // Compare again but with debug info
    compare_net_outputs(policy_data, cpu_policy_data, fatal, true, "orig policy");
    compare_net_outputs(value_data, cpu_value_data, fatal, true, "orig value");
    // Call opencl.forward again to see if the error is reproducible.
    std::vector<float> value_data_retry(value_data.size());
    std::vector<float> policy_data_retry(policy_data.size());
    opencl_.forward(input_data, policy_data_retry, value_data_retry);
    bool almost_equal_retry = compare_net_outputs(policy_data_retry, policy_data, fatal, true, "retry policy");
    almost_equal_retry &= compare_net_outputs(value_data_retry, value_data, fatal, true, "retry value");
    if (!almost_equal_retry) {
      throw std::runtime_error("OpenCL retry self-check mismatch.");
    } else {
      printf("compare_net_outputs retry was ok\n");
    }
    if (fatal) {
      printf("Update your GPU drivers or reduce the amount of games played simultaneously.\n");
      throw std::runtime_error("OpenCL self-check mismatch.");
    }
  }
}

REGISTER_NETWORK("opencl", OpenCLNetwork, 100)

} // namespace lczero
