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

#include <condition_variable>
#include <queue>
#include <thread>
#include "utils/exception.h"

namespace lczero {
namespace {

class MuxingNetwork;
class MuxingComputation : public NetworkComputation {
 public:
  MuxingComputation(MuxingNetwork* network) : network_(network) {}

  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return planes_.size(); }

  float GetQVal(int sample) const override {
    return parent_->GetQVal(sample + idx_in_parent_);
  }

  float GetPVal(int sample, int move_id) const override {
    return parent_->GetPVal(sample + idx_in_parent_, move_id);
  }

  void PopulateToParent(std::shared_ptr<NetworkComputation> parent) {
    // Populate our batch into batch of batches.
    parent_ = parent;
    idx_in_parent_ = parent->GetBatchSize();
    for (auto& x : planes_) parent_->AddInput(std::move(x));
  }

  void NotifyReady() { dataready_cv_.notify_one(); }

 private:
  std::vector<InputPlanes> planes_;
  MuxingNetwork* network_;
  std::shared_ptr<NetworkComputation> parent_;
  int idx_in_parent_ = 0;

  std::condition_variable dataready_cv_;
};

class MuxingNetwork : public Network {
 public:
  MuxingNetwork(const Weights& weights, const OptionsDict& options) {
    // int threads, int max_batch)
    //: network_(std::move(network)), max_batch_(max_batch) {

    const auto parents = options.ListSubdicts();
    if (parents.empty()) {
      throw Exception("Empty list of backends passed to a Muxing backend");
    }

    for (const auto& name : parents) {
      const auto& opts = options.GetSubdict(name);
      const int nn_threads = opts.GetOrDefault<int>("threads", 1);
      const int max_batch = opts.GetOrDefault<int>("max_batch", 256);
      const std::string backend =
          opts.GetOrDefault<std::string>("backend", name);

      networks_.emplace_back(
          NetworkFactory::Get()->Create(backend, weights, opts));
      Network* net = networks_.back().get();

      for (int i = 0; i < nn_threads; ++i) {
        threads_.emplace_back(
            [this, net, max_batch]() { Worker(net, max_batch); });
      }
    }
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<MuxingComputation>(this);
  }

  void Enqueue(MuxingComputation* computation) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(computation);
    cv_.notify_one();
  }

  ~MuxingNetwork() {
    Abort();
    Wait();
    // Unstuck waiting computations.
    while (!queue_.empty()) {
      queue_.front()->NotifyReady();
      queue_.pop();
    }
  }

  void Worker(Network* network, const int max_batch) {
    // While Abort() is not called (and it can only be called from destructor).
    while (!abort_) {
      std::vector<MuxingComputation*> children;
      // Create new computation in "upstream" network, to gather batch into
      // there.
      std::shared_ptr<NetworkComputation> parent(network->NewComputation());
      {
        std::unique_lock<std::mutex> lock(mutex_);
        // Wait until there's come work to compute.
        cv_.wait(lock, [&] { return abort_ || !queue_.empty(); });
        if (abort_) break;

        // While there is a work in queue, add it.
        while (!queue_.empty()) {
          // If we are reaching batch size limit, stop adding.
          // However, if a single input batch is larger than output batch limit,
          // we still have to add it.
          if (parent->GetBatchSize() != 0 &&
              parent->GetBatchSize() + queue_.front()->GetBatchSize() >
                  max_batch) {
            break;
          }
          // Remember which of "input" computations we serve.
          children.push_back(queue_.front());
          queue_.pop();
          // Make "input" computation populate data into output batch.
          children.back()->PopulateToParent(parent);
        }
      }

      // Compute.
      parent->ComputeBlocking();
      // Notify children that data is ready!
      for (auto child : children) child->NotifyReady();
    }
  }

  void Abort() {
    std::lock_guard<std::mutex> lock(mutex_);
    abort_ = true;
    cv_.notify_all();
  }

  void Wait() {
    while (!threads_.empty()) {
      threads_.back().join();
      threads_.pop_back();
    }
  }

 private:
  std::vector<std::unique_ptr<Network>> networks_;
  std::queue<MuxingComputation*> queue_;
  bool abort_ = false;

  std::mutex mutex_;
  std::condition_variable cv_;

  std::vector<std::thread> threads_;
};

void MuxingComputation::ComputeBlocking() {
  std::mutex mx;
  std::unique_lock<std::mutex> lock(mx);
  network_->Enqueue(this);
  dataready_cv_.wait(lock);
}

}  // namespace

REGISTER_NETWORK("multiplexing", MuxingNetwork, -1);

}  // namespace lczero