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
#include <cblas.h>

namespace lczero {

class BlasNetwork : public BlasCLNetwork {
 public:
  BlasNetwork(const Weights& weights, const OptionsDict& options)
    : BlasCLNetwork(weights, options { }

  
 protected:
  void winograd_transform_in();
  void winograd_sgemm();
  void winograd_transform_out();
  void winograd_convolve3();
  template<unsigned int filter_size> void convolve();
  template<size_t spatial_size> void batchnorm();
}

REGISTER_NETWORK("blas", BlasNetwork, 80);

} // namespace lczero
