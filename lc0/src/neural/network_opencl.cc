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

#include "neural/factory.h"
#include "utils/bititer.h"
#include "utils/optionsdict.h"
#include "utils/transpose.h"

namespace lczero {

namespace {


class OCLNetworkComputation;
class OCLNetwork : public Network {
 public:
  OCLNetwork(const Weights& weights, const OptionsDict& options);

  std::unique_ptr<NetworkComputation> NewComputation() override;

 //private:
  //tensorflow::Scope scope_;
  //tensorflow::ClientSession session_;

  //std::unique_ptr<tensorflow::ops::Placeholder> input_;
  //std::unique_ptr<tensorflow::Output> policy_head_;
  //std::unique_ptr<tensorflow::Output> value_head_;
};

class OCLNetworkComputation : public NetworkComputation {
 public:
  OCLNetworkComputation(const OCLNetwork* network) : network_(network) {}
  void AddInput(InputPlanes&& input) override {
    batchsize_++;
    printf("OCL AddInput %d\n", batchsize_);
  }
  void ComputeBlocking() override {
    batchsize_--;
    printf("OCL ComputeBlocking %d\n", batchsize_);
  }

  int GetBatchSize() const override { 
    printf("OCL GetBatchSize: %d\n", batchsize_);
    return batchsize_;
  }
  float GetQVal(int sample) const override {
    return 0.0f;
  }
  float GetPVal(int sample, int move_id) const override {
    return 0.1f; // TODO: Maybe it has to be normalized to 100% by now?
  }

 private:
  const OCLNetwork* network_;
  int batchsize_ = 0;
};

OCLNetwork::OCLNetwork(const Weights& weights, const OptionsDict& options) {
  printf("OCLNetwork construct\n");
}

std::unique_ptr<NetworkComputation> OCLNetwork::NewComputation() {
  return std::make_unique<OCLNetworkComputation>(this);
}

}  // namespace

// TODO: Pick priority. Guessing it should be lower than TF?
// Or is TF CPU only, and OpenCL would be better?
// REGISTER_NETWORK("opencl", OCLNetwork, 90);
REGISTER_NETWORK("opencl", OCLNetwork, 999);

}  // namespace lczero
