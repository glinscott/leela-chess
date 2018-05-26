/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <cstddef>
#include "network.h"

/* Features */
#if defined(USE_BLAS) && !defined(__APPLE__) && !defined(__MACOSX)
#define USE_OPENBLAS // TODO: move this to meson
#endif
//#define USE_MKL

//#ifndef FEATURE_USE_CPU_ONLY
//#define USE_OPENCL
//#define USE_OPENCL_SELFCHECK
//#endif
//#define USE_TUNER

// OpenBLAS limitation
#if defined(USE_BLAS) && defined(USE_OPENBLAS)
#define MAX_CPUS 64
#else
#define MAX_CPUS 128
#endif

typedef float net_t; // TODO: move this to network.h, use Vec everywhere

size_t ceilMultiple(size_t a, size_t b);

static constexpr auto WINOGRAD_ALPHA = 4;
static constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
static constexpr auto WINOGRAD_P = 8 * 8 / 4;


namespace lczero { // TODO: The stuff above is used by OpenCL

  class BlasTransforms {
   public:
    static void initOneBlock(Weights::ConvBlock& block, bool inputlayer=false, bool headlayer=false);

    static std::vector<float> winograd_transform_f(const std::vector<float>& f,
                                                   const int outputs,
                                                   const int channels);

    static void winograd_transform_in(const std::vector<float>& in,
                                      std::vector<float>& V,
                                      const int C);

    static void winograd_sgemm(const std::vector<float>& U,
                               std::vector<float>& V,
                               std::vector<float>& M,
                               const int C, const int K);

    static void winograd_transform_out(const std::vector<float>& M,
                                       std::vector<float>& Y,
                                       const int K);

    static void winograd_convolve3(const int outputs,
                                   const std::vector<float>& input,
                                   const std::vector<float>& U,
                                   std::vector<float>& V,
                                   std::vector<float>& M,
                                   std::vector<float>& output);

    static void convolve(size_t outputs,
                         const std::vector<float>& input,
                         const std::vector<float>& weights,
                         const std::vector<float>& biases,
                         std::vector<float>& output);

    static void batchnorm(size_t channels,
                          std::vector<float>& data,
                          const std::vector<float>& means,
                          const std::vector<float>& stddivs,
                          const float* eltwise = nullptr);

    static void innerproduct(const std::vector<float>& inputs,
                             const std::vector<float>& weights,
                             const std::vector<float>& biases,
                             std::vector<float>& outputs,
                             bool apply_relu=false);

    static std::vector<float> softmax(const std::vector<float>& input, float temperature=1.0f);
    // TODO: softmaxtemp hardcoded from lczero

    static std::vector<float> zeropad_U(const std::vector<float>& U,
                                        const int outputs, const int channels,
                                        const int outputs_pad,
                                        const int channels_pad);
  };

} // namespace lczero
