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

#include "network_blas_cl_common.cc" // ???
#include "OpenCLScheduler.h"
#include <atomic>

namespace lczero {

class OpenCLNetwork : public BlasNetwork {
 public:
  OpenCLNetwork(const Weights& weights, const OptionsDict& options)
    : BlasNetwork(weights, options) {
    //myprintf("Initializing OpenCL.\n");
    opencl.initialize(weights.input.biases.size());
    // etc, tbd (refactoring necessary?)
  }
  std::pair<float value, float[] policy> forward(InputPlanes&& input) override;

 protected:
  void initOneBlock(Weights::ConvBlock& block, bool inputlayer=false) override;
}


REGISTER_NETWORK("opencl", OpenCLNetwork, 90);

} // namespace lczero
