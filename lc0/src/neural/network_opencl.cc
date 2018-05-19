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
#include "neural/network_old.h"
#include <cassert>

namespace lczero {

namespace {


class OpenCLNetworkComputation;
class OpenCLNetwork : public Network {
 public:
  OpenCLNetwork(const Weights& weights, const OptionsDict& options);

  std::unique_ptr<NetworkComputation> NewComputation() override;

 //private:
  //tensorflow::Scope scope_;
  //tensorflow::ClientSession session_;

  //std::unique_ptr<tensorflow::ops::Placeholder> input_;
  //std::unique_ptr<tensorflow::Output> policy_head_;
  //std::unique_ptr<tensorflow::Output> value_head_;
};

class OpenCLNetworkComputation : public NetworkComputation {
 public:
  OpenCLNetworkComputation(const OpenCLNetwork* network) : network_(network) {}
  void AddInput(InputPlanes&& input) override {
    raw_input_.emplace_back(input);
    printf("OpenCL AddInput %ld\n", raw_input_.size());
  }
  void ComputeBlocking() override {
    // TODO: Should we do FIFO, not stack?
    // I think it doesn't matter because with batch_size=1
    // raw_input_ will only get one value per instance of this object.
    ::Network::get_scored_moves(raw_input_.back());
    printf("OpenCL ComputeBlocking %ld\n", raw_input_.size());
  }

  int GetBatchSize() const override { 
    printf("OpenCL GetBatchSize: %ld\n", raw_input_.size());
    assert(raw_input_.size() <= 1);
    return raw_input_.size();
  }
  float GetQVal(int sample) const override {
    (void)sample; // TODO
    return 0.0f;
  }
  float GetPVal(int sample, int move_id) const override {
    (void)sample; // TODO
    (void)move_id; // TODO
    return 0.1f; // TODO: Maybe it has to be normalized to 100% by now?
  }

 private:
  const OpenCLNetwork* network_;
  std::vector<InputPlanes> raw_input_;
};

OpenCLNetwork::OpenCLNetwork(const Weights& weights, const OptionsDict& options) {
  (void)weights; // TODO
  (void)options; // TODO
  printf("OpenCLNetwork construct\n");
  ::Network::initialize();
}

std::unique_ptr<NetworkComputation> OpenCLNetwork::NewComputation() {
  return std::make_unique<OpenCLNetworkComputation>(this);
}

}  // namespace

// TODO: Pick priority. Guessing it should be lower than TF?
// Or is TF CPU only, and OpenCL would be better?
// REGISTER_NETWORK("opencl", OpenCLNetwork, 90)
REGISTER_NETWORK("opencl", OpenCLNetwork, 999)

}  // namespace lczero
