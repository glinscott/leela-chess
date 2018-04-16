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
    static constexpr int MAX_FORMAT_VERSION = 2;
    static constexpr int T_HISTORY = 8;

    // Version 1:
    // 120 input channels
    // 14 * (6 us, 6 them, 2 reps) = 112
    // 4 castling (us_oo, us_ooo, them_oo, them_ooo)
    // 1 color
    // 1 rule50_count
    // 1 move_count
    // 1 unused plane of zeros at end to pad it out.

    // Version 2:
    // 120 input channels
    // 13 * (6 us, 6 them, 1 reps) = 104
    // 1 rule50_count
    // 4 castling (us_oo, us_ooo, them_oo, them_ooo)
    // 1 color
    // 1 move_count
    // 1 unused plane of zeros? ones? at end to pad it out.

    // TODO: Should we send ones for padding?

    static constexpr int V1_HIST_PLANES = 14;
    static constexpr int V2_HIST_PLANES = 13;
    static constexpr int V1_INPUT_CHANNELS = 8 + V1_HIST_PLANES * T_HISTORY;
    static constexpr int V2_INPUT_CHANNELS = 8 + V2_HIST_PLANES * T_HISTORY;
    static constexpr int MAX_INPUT_CHANNELS = std::max(V1_INPUT_CHANNELS, V2_INPUT_CHANNELS);

    static constexpr int V1_NUM_OUTPUT_POLICY = 1924;
    static constexpr int V2_NUM_OUTPUT_POLICY = 1858;
    static constexpr int NUM_OUTPUT_VALUE = 1;
    static constexpr int NUM_VALUE_CHANNELS = 128;
    static constexpr int NUM_VALUE_INPUT_PLANES = 32;
    static constexpr int NUM_POLICY_INPUT_PLANES = 32;

    using scored_node = std::pair<float, Move>;
    using Netresult = std::pair<std::vector<scored_node>, float>;
    using BoardPlane = std::bitset<8 * 8>;

    struct NNPlanes {
      // For V2 some of this is unused
      std::array<BoardPlane, MAX_INPUT_CHANNELS - 3> bit;
      int rule50_count;
      int move_count;
    };

    struct DebugRawData {
      std::vector<float> input;
      std::vector<float> policy_output;
      float value_output;
      std::vector<scored_node> filtered_output;

      std::string getJson() const;
    };

    static Netresult get_scored_moves(const BoardHistory& state,
                                      DebugRawData* debug_data=nullptr,
                                      bool skip_cache = false);

    // Winograd filter transformation changes 3x3 filters to 4x4
    static constexpr auto WINOGRAD_ALPHA = 4;
    static constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;

    static void initialize();

    static int lookup(Move move, Color c);
    static void gather_features(const BoardHistory& pos, NNPlanes& planes);
    static size_t get_format_version();
    static size_t get_input_channels();
    static size_t get_hist_planes();
    static size_t get_num_output_policy();

private:
    static bool initialized;
    static std::pair<int, int> load_network(std::istream& wtfile);
    static std::pair<int, int> load_network_file(std::string filename);
    static void process_bn_var(std::vector<float>& weights,
                               const float epsilon=1e-5f);
    static size_t m_format_version;
    static std::unordered_map<Move, int, std::hash<int>> old_move_lookup;
    static std::unordered_map<Move, int, std::hash<int>> new_move_lookup;

    static void softmax(const std::vector<float>& input,
                        std::vector<float>& output,
                        float temperature = 1.0f);
    static std::vector<float> winograd_transform_f(const std::vector<float>& f,
        const int outputs, const int channels);
    static std::vector<float> zeropad_U(const std::vector<float>& U,
        const int outputs, const int channels,
        const int outputs_pad, const int channels_pad);
    static void winograd_transform_in(const std::vector<float>& in,
                                      std::vector<float>& V,
                                      const int C);
    static void winograd_transform_out(const std::vector<float>& M,
                                       std::vector<float>& Y,
                                       const int K);
    static void winograd_convolve3(const int outputs,
                                   const std::vector<float>& input,
                                   const std::vector<float>& U,
                                   std::vector<float>& V,
                                   std::vector<float>& M,
                                   std::vector<float>& output);
    static void winograd_sgemm(const std::vector<float>& U,
                               std::vector<float>& V,
                               std::vector<float>& M, const int C, const int K);
    static void init_move_map();
    static Netresult get_scored_moves_internal(const BoardHistory& state, NNPlanes& planes, DebugRawData* debug_data);
#if defined(USE_BLAS)
    static void forward_cpu(std::vector<float>& input,
                            std::vector<float>& output_pol,
                            std::vector<float>& output_val);

#endif
};

#endif
