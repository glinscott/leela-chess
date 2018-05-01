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

#include <memory>
#include <vector>

namespace lczero {

const int kInputPlanes = 112;

struct Weights {
  using Vec = std::vector<float>;
  struct ConvBlock {
    Vec weights;
    Vec biases;
    Vec bn_means;
    Vec bn_stddivs;
  };

  struct Residual {
    ConvBlock conv1;
    ConvBlock conv2;
  };

  // Input convnet.
  ConvBlock input;

  // Residual tower.
  std::vector<Residual> residual;

  // Policy head
  ConvBlock policy;
  Vec ip_pol_w;
  Vec ip_pol_b;

  // Value head
  ConvBlock value;
  Vec ip1_val_w;
  Vec ip1_val_b;
  Vec ip2_val_w;
  Vec ip2_val_b;
};

// All input planes are 64 value vectors, every element of which is either
// 0 or some value, unique for the plane. Therefore, input is defined as
// a bitmask showing where to set the value, and the value itself.
struct InputPlane {
  InputPlane() = default;
  void SetAll() { mask = ~0ull; }
  void Fill(float val) {
    SetAll();
    value = val;
  }
  std::uint64_t mask = 0ull;
  float value = 1.0f;
};
using InputPlanes = std::vector<InputPlane>;

// An interface to implement by computing backends.
class NetworkComputation {
 public:
  // Adds a sample to the batch.
  virtual void AddInput(InputPlanes&& input) = 0;
  // Do the computation.
  virtual void ComputeBlocking() = 0;
  // Returns how many times AddInput() was called.
  virtual int GetBatchSize() const = 0;
  // Returns Q value of @sample.
  virtual float GetQVal(int sample) const = 0;
  // Returns P value @move_id of @sample.
  virtual float GetPVal(int sample, int move_id) const = 0;
  virtual ~NetworkComputation() {}
};

class Network {
 public:
  virtual std::unique_ptr<NetworkComputation> NewComputation() = 0;
  virtual ~Network(){};
};

}  // namespace lczero