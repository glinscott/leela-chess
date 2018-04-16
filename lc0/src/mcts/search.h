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

#include <functional>
#include <shared_mutex>
#include <thread>
#include "mcts/node.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "uciloop.h"
#include "ucioptions.h"

namespace lczero {

struct SearchLimits {
  std::int64_t nodes = -1;
  std::int64_t time_ms = -1;
};

class Search {
 public:
  Search(Node* root_node, NodePool* node_pool, const Network* network,
         BestMoveInfo::Callback best_move_callback,
         UciInfo::Callback info_callback, const SearchLimits& limits,
         UciOptions* uci_options, NNCache* cache);

  ~Search();

  // Populates UciOptions with search parameters.
  static void PopulateUciParams(UciOptions* options);

  // Starts worker threads and returns immediately.
  void StartThreads(int how_many);

  // Stops search. At the end bestmove will be returned. The function is not
  // blocking, so it returns before search is actually done.
  void Stop();
  // Stops search, but does not return bestmove. The function is not blocking.
  void Abort();
  // Aborts the search, and blocks until all worker thread finish.
  void AbortAndWait();

  // Returns best move, from the point of view of white player. And also ponder.
  std::pair<Move, Move> GetBestMove() const;

 private:
  // Can run several copies of it in separate threads.
  void Worker();

  uint64_t GetTimeSinceStart() const;
  void MaybeTriggerStop();
  void MaybeOutputInfo();
  bool AddNodeToCompute(Node* node, CachingComputation* computation,
                        bool add_if_cached = true);
  int PrefetchIntoCache(Node* node, int budget,
                        CachingComputation* computation);

  void SendUciInfo();  // Requires nodes_mutex_ to be held.

  Node* PickNodeToExtend(Node* node);
  InputPlanes EncodeNode(const Node* node);
  void ExtendNode(Node* node);

  std::mutex counters_mutex_;
  bool stop_ = false;
  bool responded_bestmove_ = false;
  std::vector<std::thread> threads_;

  Node* root_node_;
  NodePool* node_pool_;
  NNCache* cache_;

  mutable std::shared_mutex nodes_mutex_;
  const Network* network_;
  const SearchLimits limits_;
  const std::chrono::steady_clock::time_point start_time_;
  Node* best_move_node_ = nullptr;
  Node* last_outputted_best_move_node_ = nullptr;
  UciInfo uci_info_;
  uint64_t total_nodes_ = 0;

  BestMoveInfo::Callback best_move_callback_;
  UciInfo::Callback info_callback_;

  // External parameters.
  const int kMiniBatchSize;
  const int kMiniPrefetchBatch;
  const bool kAggresiveCaching;
  const float kCpuct;
};

}  // namespace lczero