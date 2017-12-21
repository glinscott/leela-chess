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

#ifdef USE_OPENCL
#include <atomic>
class UCTNode;
#endif

#include "Position.h"

class Network {
public:
    using BoardPlane = std::bitset<8*8>;
    using NNPlanes = std::vector<BoardPlane>;
    using scored_node = std::pair<float, int>;
    using Netresult = std::pair<std::vector<scored_node>, float>;

    static Netresult get_scored_moves(Position* state);
    // File format version
    static constexpr int FORMAT_VERSION = 1;
    static constexpr int T_HISTORY = 8;
    static constexpr int INPUT_CHANNELS = 5 + T_HISTORY*14;  //--ignoring the halfmove and fullmove clocks for now.

    static void init();
//    static void benchmark(Position* state, int iterations = 1600);
    static void show_heatmap(Position* state, Netresult& netres, bool topmoves);
    static void softmax(const std::vector<float>& input, std::vector<float>& output, float temperature = 1.0f);
    static void gather_features(Position* pos, NNPlanes& planes);

private:
    static Netresult get_scored_moves_internal(Position* state, NNPlanes& planes);
};

#endif
