/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "config.h"
#include <vector>
#include <string>
#include <bitset>
#include <memory>
#include <array>
#include <unordered_map>

#ifdef USE_OPENCL
#include <atomic>
class UCTNode;
#endif

#include "Position.h"

class Network {
public:
    // File format version
    static constexpr int FORMAT_VERSION = 1;
    static constexpr int T_HISTORY = 8;

    // 120 input channels
    // 14 * (6 us, 6 them, 2 reps)
    // 4 castling (us_oo, us_ooo, them_oo, them_ooo)
    // 1 color
    // 1 rule50_count
    // 1 move_count
    // 1 unused at end to pad it out.
    static constexpr int INPUT_CHANNELS = 8 + 14 * T_HISTORY;

    static constexpr int NUM_OUTPUT_POLICY = 1924;
    static constexpr int NUM_VALUE_CHANNELS = 256;

    using BoardPlane = std::bitset<8 * 8>;
    struct NNPlanes {
      std::array<BoardPlane, INPUT_CHANNELS - 3> bit;
      int rule50_count;
      int move_count;
    };

    using scored_node = std::pair<float, Move>;
    using Netresult = std::pair<std::vector<scored_node>, float>;

    struct DebugRawData {
      std::vector<float> input;
      std::vector<float> policy_output;
      float value_output;
      std::vector<scored_node> filtered_output;

      std::string getJson() const;
    };

    static std::unordered_map<Move, int, std::hash<int>> move_lookup;
    static Netresult get_scored_moves(const BoardHistory& state, DebugRawData* debug_data=nullptr);

    static void init();
//    static void benchmark(Position* state, int iterations = 1600);
    static void show_heatmap(const BoardHistory& state, Netresult& netres, bool topmoves);
    static void softmax(const std::vector<float>& input, std::vector<float>& output, float temperature = 1.0f);
    static void gather_features(const BoardHistory& pos, NNPlanes& planes);

private:
    static void process_bn_var(std::vector<float>& weights, const float epsilon=1e-5f);
    static void init_move_map();
    static Netresult get_scored_moves_internal(const BoardHistory& state, NNPlanes& planes, DebugRawData* debug_data);
#if defined(USE_BLAS)
    static void forward_cpu(std::vector<float>& input,
                            std::vector<float>& output);
#endif
};

#endif
