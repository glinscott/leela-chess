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

#include "config.h"
#include "Network.h"
#include "UCI.h"

#include <algorithm>
#include <cassert>
#include <stack>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>
#include <memory>
#include <cmath>
#include <array>
#include <thread>
#include <boost/utility.hpp>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>
#include "zlib.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#include "UCTNode.h"
#endif

#include "Random.h"
#include "Network.h"
#include "NNCache.h"
#include "Utils.h"
#include "Parameters.h"
#include "Timing.h"
#include "Movegen.h"
#include "ThreadPool.h"
#include "Im2Col.h"

namespace x3 = boost::spirit::x3;
using namespace Utils;

// Seems this is required for the Darwin compiler.
// Not sure why only T_HISTORY and not all the others.
constexpr int Network::T_HISTORY;

bool Network::initialized = false;
size_t Network::m_format_version{0};
std::unordered_map<Move, int, std::hash<int>> Network::old_move_lookup;
std::unordered_map<Move, int, std::hash<int>> Network::new_move_lookup;

// Input + residual block tower
static std::vector<std::vector<float>> conv_weights;
static std::vector<std::vector<float>> conv_biases;
static std::vector<std::vector<float>> batchnorm_means;
static std::vector<std::vector<float>> batchnorm_stddivs;

// Policy head
static std::vector<float> conv_pol_w;
static std::vector<float> conv_pol_b;
static std::array<float, Network::NUM_POLICY_INPUT_PLANES> bn_pol_w1;
static std::array<float, Network::NUM_POLICY_INPUT_PLANES> bn_pol_w2;

// TODO: These are compile time sized,
// It would be nicer to dynamically size.
// Just a little memory optimization.
// But maybe there is a reason they must be array not vector?
static std::array<float, Network::V1_NUM_OUTPUT_POLICY*8*8*Network::NUM_POLICY_INPUT_PLANES> v1_ip_pol_w;
static std::array<float, Network::V1_NUM_OUTPUT_POLICY> v1_ip_pol_b;
static std::array<float, Network::V2_NUM_OUTPUT_POLICY*8*8*Network::NUM_POLICY_INPUT_PLANES> v2_ip_pol_w;
static std::array<float, Network::V2_NUM_OUTPUT_POLICY> v2_ip_pol_b;

// Value head
static std::vector<float> conv_val_w;
static std::vector<float> conv_val_b;
static std::array<float, Network::NUM_VALUE_INPUT_PLANES> bn_val_w1;
static std::array<float, Network::NUM_VALUE_INPUT_PLANES> bn_val_w2;

static std::array<float, Network::NUM_VALUE_CHANNELS*8*8*Network::NUM_VALUE_INPUT_PLANES> ip1_val_w;
static std::array<float, Network::NUM_VALUE_CHANNELS> ip1_val_b;

static std::array<float, Network::NUM_VALUE_CHANNELS> ip2_val_w;
static std::array<float, 1> ip2_val_b;

size_t Network::get_format_version() {
    return m_format_version;
}

size_t Network::get_input_channels() {
    return m_format_version == 1 ? V1_INPUT_CHANNELS : V2_INPUT_CHANNELS;
}

size_t Network::get_hist_planes() {
    return m_format_version == 1 ? V1_HIST_PLANES : V2_HIST_PLANES;
}

size_t Network::get_num_output_policy() {
    return m_format_version == 1 ? V1_NUM_OUTPUT_POLICY : V2_NUM_OUTPUT_POLICY;
}

void Network::process_bn_var(std::vector<float>& weights, const float epsilon) {
    for(auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

std::vector<float> Network::winograd_transform_f(const std::vector<float>& f,
                                                 const int outputs,
                                                 const int channels) {
    // F(2x2, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    auto G = std::array<float, WINOGRAD_TILE>{ 1.0,  0.0,  0.0,
                                               0.5,  0.5,  0.5,
                                               0.5, -0.5,  0.5,
                                               0.0,  0.0,  1.0};
    auto temp = std::array<float, 12>{};

    for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
            for (auto i = 0; i < 4; i++){
                for (auto j = 0; j < 3; j++) {
                    auto acc = 0.0f;
                    for (auto k = 0; k < 3; k++) {
                        acc += G[i*3 + k] * f[o*channels*9 + c*9 + k*3 + j];
                    }
                    temp[i*3 + j] = acc;
                }
            }

            for (auto xi = 0; xi < 4; xi++) {
                for (auto nu = 0; nu < 4; nu++) {
                    auto acc = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        acc += temp[xi*3 + k] * G[nu*3 + k];
                    }
                    U[xi * (4 * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o] = acc;
                }
            }
        }
    }

    return U;
}

std::vector<float> Network::zeropad_U(const std::vector<float>& U,
                                      const int outputs, const int channels,
                                      const int outputs_pad,
                                      const int channels_pad) {
    // Fill with zeroes
    auto Upad = std::vector<float>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for(auto o = 0; o < outputs; o++) {
        for(auto c = 0; c < channels; c++) {
            for(auto xi = 0; xi < WINOGRAD_ALPHA; xi++){
                for(auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                    Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad)
                         + c * outputs_pad +
                          o] =
                    U[xi * (WINOGRAD_ALPHA * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o];
                }
            }
        }
    }

    return Upad;
}

extern "C" void openblas_set_num_threads(int num_threads);

std::pair<int, int> Network::load_network(std::istream& wtfile) {
    // Read format version
    auto line = std::string{};
    if (std::getline(wtfile, line)) {
        auto iss = std::stringstream{ line };
        // First line is the file format version id
        iss >> m_format_version;
        if (iss.fail()
            || m_format_version > MAX_FORMAT_VERSION
            || m_format_version < 1) {
            myprintf("Weights file is the wrong version.\n");
            return {0, 0};
        } else {
            assert(m_format_version <= MAX_FORMAT_VERSION);
        }
    } else {
        myprintf("Weights file is empty.\n");
        return {0, 0};
    }
    // Count size of the network
    myprintf("Detecting residual layers...");
    myprintf("v%d...", m_format_version);
    // First line was the version number
    auto linecount = size_t{1};
    auto channels = 0;
    while (std::getline(wtfile, line)) {
        auto iss = std::stringstream{line};
        // Third line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // We are assuming all layers have the same amount of filters.
        if (linecount == 2) {
            auto count = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            myprintf("%d channels...", count);
            channels = count;
        }
        linecount++;
    }
    // 1 format id, 1 input layer (4 x weights), 14 ending weights,
    // the rest are residuals, every residual has 8 x weight lines
    // Note: 14 ending weights is for value/policy head.
    //     It's a coincidence it's the same number of input features
    //     for V1 networks.
    auto residual_blocks = linecount - (1 + 4 + 14);
    if (residual_blocks % 8 != 0) {
        myprintf("\nInconsistent number of weights in the file.\n");
        myprintf("%d %d %d %d\n", m_format_version, residual_blocks, linecount, get_hist_planes());
        return {0, 0};
    }
    residual_blocks /= 8;
    myprintf("%d blocks.\n", residual_blocks);

    // Re-read file and process
    wtfile.clear();
    wtfile.seekg(0, std::ios::beg);

    // Get the file format id out of the way
    std::getline(wtfile, line);

    auto plain_conv_layers = 1 + (residual_blocks * 2);
    auto plain_conv_wts = plain_conv_layers * 4;
    linecount = 0;
    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        auto it_line = cbegin(line);
        const auto ok = phrase_parse(it_line, cend(line),
                                     *x3::float_, x3::space, weights);
        if (!ok || it_line != cend(line)) {
            myprintf("\nFailed to parse weight file. Error on line %d.\n",
                    linecount + 2); //+1 from version line, +1 from 0-indexing
            return {0, 0};
        }
        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
                // Redundant in our model, but they encode the
                // number of outputs so we have to read them in.
                conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
                batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
                process_bn_var(weights);
                batchnorm_stddivs.emplace_back(weights);
            }
        } else if (linecount == plain_conv_wts) {
            conv_pol_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 1) {
            conv_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 2) {
            std::copy(begin(weights), end(weights), begin(bn_pol_w1));
        } else if (linecount == plain_conv_wts + 3) {
            process_bn_var(weights);
            std::copy(begin(weights), end(weights), begin(bn_pol_w2));
        } else if (linecount == plain_conv_wts + 4) {
            if (m_format_version == 1) {
                std::copy(begin(weights), end(weights), begin(v1_ip_pol_w));
            } else {
                std::copy(begin(weights), end(weights), begin(v2_ip_pol_w));
            }
        } else if (linecount == plain_conv_wts + 5) {
            if (m_format_version == 1) {
                std::copy(begin(weights), end(weights), begin(v1_ip_pol_b));
            } else {
                std::copy(begin(weights), end(weights), begin(v2_ip_pol_b));
            }
        } else if (linecount == plain_conv_wts + 6) {
            conv_val_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 7) {
            conv_val_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 8) {
            std::copy(begin(weights), end(weights), begin(bn_val_w1));
        } else if (linecount == plain_conv_wts + 9) {
            process_bn_var(weights);
            std::copy(begin(weights), end(weights), begin(bn_val_w2));
        } else if (linecount == plain_conv_wts + 10) {
            std::copy(begin(weights), end(weights), begin(ip1_val_w));
        } else if (linecount == plain_conv_wts + 11) {
            std::copy(begin(weights), end(weights), begin(ip1_val_b));
        } else if (linecount == plain_conv_wts + 12) {
            std::copy(begin(weights), end(weights), begin(ip2_val_w));
        } else if (linecount == plain_conv_wts + 13) {
            std::copy(begin(weights), end(weights), begin(ip2_val_b));
        }
        linecount++;
    }

    return {channels, residual_blocks};
}

std::pair<int, int> Network::load_network_file(std::string filename) {
    // gzopen supports both gz and non-gz files, will decompress or just read directly as needed.
    auto gzhandle = gzopen(filename.c_str(), "rb");
    if (gzhandle == nullptr) {
        myprintf("Could not open weights file: %s\n", filename.c_str());
        return {0, 0};
    }
    // Stream the gz file in to a memory buffer stream.
    std::stringstream buffer;
    const int chunkBufferSize = 64 * 1024;
    std::vector<char> chunkBuffer(chunkBufferSize);
    while (true) {
        int bytesRead = gzread(gzhandle, chunkBuffer.data(), chunkBufferSize);
        if (bytesRead == 0) break;
        if (bytesRead < 0) {
            myprintf("Failed to decompress or read: %s\n", filename.c_str());
            gzclose(gzhandle);
            return {0, 0};
        }
        assert(bytesRead <= chunkBufferSize);
        buffer.write(chunkBuffer.data(), bytesRead);
    }
    auto result = load_network(buffer);
    gzclose(gzhandle);
    return result;
}

void Network::initialize(void) {
    if (initialized) return;
    initialized = true;

    init_move_map();

    // Load network from file
    size_t channels, residual_blocks;
    assert(m_format_version == 0);
    std::tie(channels, residual_blocks) = load_network_file(cfg_weightsfile);
    assert(m_format_version > 0);
    if (channels == 0) {
        exit(EXIT_FAILURE);
    }

    auto weight_index = size_t{0};
    // Input convolution
    // Winograd transform convolution weights
    conv_weights[weight_index] =
        winograd_transform_f(conv_weights[weight_index],
                             channels, get_input_channels());
    weight_index++;

    // Residual block convolutions
    for (auto i = size_t{0}; i < residual_blocks * 2; i++) {
		conv_weights[weight_index] =
            winograd_transform_f(conv_weights[weight_index],
                                 channels, channels);
        weight_index++;
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    for (auto i = size_t{0}; i < conv_biases.size(); i++) {
        for (auto j = size_t{0}; j < batchnorm_means[i].size(); j++) {
            batchnorm_means[i][j] -= conv_biases[i][j];
            conv_biases[i][j] = 0.0f;
        }
    }

    if ((bn_val_w1.size() != conv_val_b.size()) ||
        (bn_pol_w1.size() != conv_pol_b.size()) ) {
            throw std::runtime_error("Weights are malformed. Incorrect number "
             "of policy/value output planes.");
    }

    for (auto i = size_t{0}; i < bn_val_w1.size(); i++) {
        bn_val_w1[i] -= conv_val_b[i];
        conv_val_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < bn_pol_w1.size(); i++) {
        bn_pol_w1[i] -= conv_pol_b[i];
        conv_pol_b[i] = 0.0f;
    }

#ifdef USE_OPENCL
    myprintf("Initializing OpenCL.\n");
    opencl.initialize(channels);

    for(auto & opencl_net : opencl.get_networks()) {
        auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        auto mwg = tuners[0];
        auto kwg = tuners[2];
        auto vwm = tuners[3];

        weight_index = 0;

        size_t m_ceil = ceilMultiple(ceilMultiple(channels, mwg), vwm);
        size_t k_ceil = ceilMultiple(ceilMultiple(get_input_channels(), kwg), vwm);

        auto Upad = zeropad_U(conv_weights[weight_index],
                              channels, get_input_channels(),
                              m_ceil, k_ceil);

        // Winograd filter transformation changes filter size to 4x4
        opencl_net->push_input_convolution(WINOGRAD_ALPHA, get_input_channels(), channels,
                Upad, batchnorm_means[weight_index], batchnorm_stddivs[weight_index]);
        weight_index++;

        // residual blocks
        for (auto i = size_t{0}; i < residual_blocks; i++) {
            auto Upad1 = zeropad_U(conv_weights[weight_index],
                                   channels, channels,
                                   m_ceil, m_ceil);
            auto Upad2 = zeropad_U(conv_weights[weight_index + 1],
                                   channels, channels,
                                   m_ceil, m_ceil);
            opencl_net->push_residual(WINOGRAD_ALPHA, channels, channels,
                                      Upad1,
                                      batchnorm_means[weight_index],
                                      batchnorm_stddivs[weight_index],
                                      Upad2,
                                      batchnorm_means[weight_index + 1],
                                      batchnorm_stddivs[weight_index + 1]);
            weight_index += 2;
        }

        // Output head convolutions
        std::vector<float> bn_pol_means(bn_pol_w1.begin(), bn_pol_w1.end());
        std::vector<float> bn_pol_stddivs(bn_pol_w2.begin(), bn_pol_w2.end());

        std::vector<float> bn_val_means(bn_val_w1.begin(), bn_val_w1.end());
        std::vector<float> bn_val_stddivs(bn_val_w2.begin(), bn_val_w2.end());

        std::vector<float> ip_pol_w_vec;
        std::vector<float> ip_pol_b_vec;
        if (m_format_version == 1) {
            ip_pol_w_vec = std::vector<float>(v1_ip_pol_w.begin(), v1_ip_pol_w.end());
            ip_pol_b_vec = std::vector<float>(v1_ip_pol_b.begin(), v1_ip_pol_b.end());
        } else {
            ip_pol_w_vec = std::vector<float>(v2_ip_pol_w.begin(), v2_ip_pol_w.end());
            ip_pol_b_vec = std::vector<float>(v2_ip_pol_b.begin(), v2_ip_pol_b.end());
        }

        std::vector<float> ip_val_w_vec(ip1_val_w.begin(), ip1_val_w.end());
        std::vector<float> ip_val_b_vec(ip1_val_b.begin(), ip1_val_b.end());

        constexpr unsigned int width = 8;
        constexpr unsigned int height = 8;

        opencl_net->push_policy(channels, NUM_POLICY_INPUT_PLANES,
                NUM_POLICY_INPUT_PLANES*width*height, get_num_output_policy(),
                conv_pol_w,
                bn_pol_means, bn_pol_stddivs,
                ip_pol_w_vec, ip_pol_b_vec);

        opencl_net->push_value(channels, NUM_VALUE_INPUT_PLANES,
                NUM_VALUE_INPUT_PLANES*width*height, NUM_VALUE_CHANNELS,
                conv_val_w,
                bn_val_means, bn_val_stddivs,
                ip_val_w_vec, ip_val_b_vec);
    }
#endif
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    myprintf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    //mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif
#endif
}

void Network::init_move_map() {
  // old includes black promotions from 2nd to 1st rank
  // new excludes those
  std::vector<Move> old_moves;
  std::vector<Move> new_moves;
  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    // Queen and knight moves
    Bitboard b = attacks_bb(QUEEN, s, 0) | attacks_bb(KNIGHT, s, 0);
    while (b) {
      old_moves.push_back(make_move(s, pop_lsb(&b)));
      new_moves.push_back(old_moves.back());
    }
  }

  // Pawn promotions
  for (Color c = WHITE; c <= BLACK; ++c) {
    for (int c_from = 0; c_from < 8; ++c_from) {
      for (int c_to = c_from - 1; c_to <= c_from + 1; ++c_to) {
        if (c_to < 0 || c_to >= 8) {
          continue;
        }
        Square from = make_square(File(c_from), c == WHITE? RANK_7 : RANK_2);
        Square to = make_square(File(c_to), c == WHITE ? RANK_8 : RANK_1);
        // push both black and white promotions for old
        old_moves.push_back(make<PROMOTION>(from, to, QUEEN));
        old_moves.push_back(make<PROMOTION>(from, to, ROOK));
        old_moves.push_back(make<PROMOTION>(from, to, BISHOP));
        // Don't need knight, as it's equivalent to pawn push to final rank.
        if (c == WHITE) {
          // only push white promotions (7th to 8th rank) for new
          new_moves.push_back(make<PROMOTION>(from, to, QUEEN));
          new_moves.push_back(make<PROMOTION>(from, to, ROOK));
          new_moves.push_back(make<PROMOTION>(from, to, BISHOP));
        }
      }
    }
  }

  for (size_t i = 0; i < old_moves.size(); ++i) {
    old_move_lookup[old_moves[i]] = i;
  }
  for (size_t i = 0; i < new_moves.size(); ++i) {
    new_move_lookup[new_moves[i]] = i;
  }
}

int Network::lookup(Move move, Color c) {
    if (type_of(move) != PROMOTION || promotion_type(move) == KNIGHT) {
        // Mask off the special move flags,
        // in particular en passant and castling
        move = Move(move & 0xfff);
    }
    if (m_format_version == 1) {
        return old_move_lookup.at(move);
    } else {
        if (c == WHITE) {
            return new_move_lookup.at(move);
        } else {
            Move flipped_move;
            if (type_of(move) == PROMOTION) {
                flipped_move = make<PROMOTION>(~from_sq(move), ~to_sq(move), promotion_type(move));
            } else {
                flipped_move = make_move(~from_sq(move), ~to_sq(move));
            }
            // The NN plays BLACK with the board flipped vertically,
            // And outputs the moves as if BLACK is moving up.
            // Flip the policy so the moves are normal.
            return new_move_lookup.at(flipped_move);
        }
    }
}

#ifdef USE_BLAS
void Network::winograd_transform_in(const std::vector<float>& in,
                                    std::vector<float>& V,
                                    const int C) {
    constexpr auto W = 8;
    constexpr auto H = 8;
    constexpr auto wtiles = (W + 1) / 2;
    constexpr auto P = wtiles * wtiles;

    for (auto ch = 0; ch < C; ch++) {
        for (auto block_y = 0; block_y < wtiles; block_y++) {
            for (auto block_x = 0; block_x < wtiles; block_x++) {

                // Tiles overlap by 2
                const auto yin = 2 * block_y - 1;
                const auto xin = 2 * block_x - 1;

                // Cache input tile and handle zero padding
                using WinogradTile =
                    std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_ALPHA>;
                WinogradTile x;

                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                        if ((yin + i) >= 0 && (xin + j) >= 0
                            && (yin + i) < H && (xin + j) < W) {
                            x[i][j] = in[ch*(W*H) + (yin+i)*W + (xin+j)];
                        } else {
                            x[i][j] = 0.0f;
                        }
                    }
                }

                const auto offset = ch*P + block_y*wtiles + block_x;

                // Calculates transpose(B).x.B
                // B = [[ 1.0,  0.0,  0.0,  0.0],
                //      [ 0.0,  1.0, -1.0,  1.0],
                //      [-1.0,  1.0,  1.0,  0.0],
                //      [ 0.0,  0.0,  0.0, -1.0]]

                WinogradTile T1, T2;

                T1[0][0] = x[0][0] - x[2][0];
                T1[0][1] = x[0][1] - x[2][1];
                T1[0][2] = x[0][2] - x[2][2];
                T1[0][3] = x[0][3] - x[2][3];
                T1[1][0] = x[1][0] + x[2][0];
                T1[1][1] = x[1][1] + x[2][1];
                T1[1][2] = x[1][2] + x[2][2];
                T1[1][3] = x[1][3] + x[2][3];
                T1[2][0] = x[2][0] - x[1][0];
                T1[2][1] = x[2][1] - x[1][1];
                T1[2][2] = x[2][2] - x[1][2];
                T1[2][3] = x[2][3] - x[1][3];
                T1[3][0] = x[1][0] - x[3][0];
                T1[3][1] = x[1][1] - x[3][1];
                T1[3][2] = x[1][2] - x[3][2];
                T1[3][3] = x[1][3] - x[3][3];

                T2[0][0] = T1[0][0] - T1[0][2];
                T2[0][1] = T1[0][1] + T1[0][2];
                T2[0][2] = T1[0][2] - T1[0][1];
                T2[0][3] = T1[0][1] - T1[0][3];
                T2[1][0] = T1[1][0] - T1[1][2];
                T2[1][1] = T1[1][1] + T1[1][2];
                T2[1][2] = T1[1][2] - T1[1][1];
                T2[1][3] = T1[1][1] - T1[1][3];
                T2[2][0] = T1[2][0] - T1[2][2];
                T2[2][1] = T1[2][1] + T1[2][2];
                T2[2][2] = T1[2][2] - T1[2][1];
                T2[2][3] = T1[2][1] - T1[2][3];
                T2[3][0] = T1[3][0] - T1[3][2];
                T2[3][1] = T1[3][1] + T1[3][2];
                T2[3][2] = T1[3][2] - T1[3][1];
                T2[3][3] = T1[3][1] - T1[3][3];

                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                        V[(i*WINOGRAD_ALPHA + j)*C*P + offset] = T2[i][j];
                    }
                }
            }
        }
    }
}

void Network::winograd_sgemm(const std::vector<float>& U,
                             std::vector<float>& V,
                             std::vector<float>& M,
                             const int C, const int K) {
    constexpr auto P = 8 * 8 / WINOGRAD_ALPHA;

    for (auto b = 0; b < WINOGRAD_TILE; b++) {
        auto offset_u = b * K * C;
        auto offset_v = b * C * P;
        auto offset_m = b * K * P;

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, P, C,
                    1.0f,
                    &U[offset_u], K,
                    &V[offset_v], P,
                    0.0f,
                    &M[offset_m], P);
    }
}

void Network::winograd_transform_out(const std::vector<float>& M,
                                     std::vector<float>& Y,
                                     const int K) {
    constexpr auto W = 8;
    constexpr auto H = 8;
    constexpr auto wtiles = (W + 1) / 2;
    constexpr auto P = wtiles * wtiles;

    for (auto k = 0; k < K; k++) {
        for (auto block_x = 0; block_x < wtiles; block_x++) {
            for (auto block_y = 0; block_y < wtiles; block_y++) {

                const auto x = 2 * block_x;
                const auto y = 2 * block_y;

                const auto b = block_y * wtiles + block_x;
                std::array<float, WINOGRAD_TILE> temp_m;
                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        temp_m[xi*WINOGRAD_ALPHA + nu] =
                            M[xi*(WINOGRAD_ALPHA*K*P) + nu*(K*P)+ k*P + b];
                    }
                }

                // Calculates transpose(A).temp_m.A
                //    A = [1.0,  0.0],
                //        [1.0,  1.0],
                //        [1.0, -1.0],
                //        [0.0, -1.0]]

                auto o11 =
                    temp_m[0*4 + 0] + temp_m[0*4 + 1] + temp_m[0*4 + 2] +
                    temp_m[1*4 + 0] + temp_m[1*4 + 1] + temp_m[1*4 + 2] +
                    temp_m[2*4 + 0] + temp_m[2*4 + 1] + temp_m[2*4 + 2];

                auto o12 =
                    temp_m[0*4 + 1] - temp_m[0*4 + 2] - temp_m[0*4 + 3] +
                    temp_m[1*4 + 1] - temp_m[1*4 + 2] - temp_m[1*4 + 3] +
                    temp_m[2*4 + 1] - temp_m[2*4 + 2] - temp_m[2*4 + 3];

                auto o21 =
                    temp_m[1*4 + 0] + temp_m[1*4 + 1] + temp_m[1*4 + 2] -
                    temp_m[2*4 + 0] - temp_m[2*4 + 1] - temp_m[2*4 + 2] -
                    temp_m[3*4 + 0] - temp_m[3*4 + 1] - temp_m[3*4 + 2];

                auto o22 =
                    temp_m[1*4 + 1] - temp_m[1*4 + 2] - temp_m[1*4 + 3] -
                    temp_m[2*4 + 1] + temp_m[2*4 + 2] + temp_m[2*4 + 3] -
                    temp_m[3*4 + 1] + temp_m[3*4 + 2] + temp_m[3*4 + 3];

                Y[k*(H*W) + (y)*W + (x)] = o11;
                if (x + 1 < W) {
                    Y[k*(H*W) + (y)*W + (x+1)] = o12;
                }
                if (y + 1 < H) {
                    Y[k*(H*W) + (y+1)*W + (x)] = o21;
                    if (x + 1 < W) {
                        Y[k*(H*W) + (y+1)*W + (x+1)] = o22;
                    }
                }
            }
        }
    }
}

void Network::winograd_convolve3(const int outputs,
                                 const std::vector<float>& input,
                                 const std::vector<float>& U,
                                 std::vector<float>& V,
                                 std::vector<float>& M,
                                 std::vector<float>& output) {

    constexpr unsigned int filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
    const auto input_channels = U.size() / (outputs * filter_len);

    winograd_transform_in(input, V, input_channels);
    winograd_sgemm(U, V, M, input_channels, outputs);
    winograd_transform_out(M, output, outputs);
}

template<unsigned int filter_size>
void convolve(size_t outputs,
              const std::vector<net_t>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    // fixed for 8x8
    constexpr unsigned int width = 8;
    constexpr unsigned int height = 8;
    constexpr unsigned int board_squares = width * height;
    constexpr unsigned int filter_len = filter_size * filter_size;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * board_squares == output.size());

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 22 3 3
    // outputs[96,8x8] = weights[96,22x3x3] x col[22x3x3,8x8]
    // C←αAB + βC
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M        N            K
                outputs, board_squares, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], board_squares,
                0.0f, &output[0], board_squares);

    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < board_squares; b++) {
            output[(o * board_squares) + b] =
                biases[o] + output[(o * board_squares) + b];
        }
    }
}

template<unsigned int inputs,
         unsigned int outputs,
         size_t W, size_t B>
void innerproduct(const std::vector<float>& input,
                  const std::array<float, W>& weights,
                  const std::array<float, B>& biases,
                  std::vector<float>& output) {
    assert(B == outputs);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);

    auto lambda_ReLU = [](float val) { return (val > 0.0f) ?
                                       val : 0.0f; };

    for (unsigned int o = 0; o < outputs; o++) {
        float val = biases[o] + output[o];
        if (outputs == Network::NUM_VALUE_CHANNELS) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }
}

template <size_t spatial_size>
void batchnorm(size_t channels,
               std::vector<float>& data,
               const float* means,
               const float* stddivs,
               const float* eltwise = nullptr)
{
    auto lambda_ReLU = [](float val) { return (val > 0.0f) ?
                                       val : 0.0f; };

    for (auto c = size_t{0}; c < channels; ++c) {
        auto mean = means[c];
        auto scale_stddiv = stddivs[c];

        if (eltwise == nullptr) {
            // Classical BN
            auto arr = &data[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(scale_stddiv * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            auto arr = &data[c * spatial_size];
            auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(res[b] +
                                     (scale_stddiv * (arr[b] - mean)));
            }
        }
    }
}

void Network::forward_cpu(std::vector<float>& input,
                          std::vector<float>& output_pol,
                          std::vector<float>& output_val) {
    // Input convolution
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int tiles = width * height / 4;
    // Calculate output channels
    const auto output_channels = conv_biases[0].size();
    //input_channels is the maximum number of input channels of any convolution.
    //Residual blocks are identical, but the first convolution might be bigger
    //when the network has very few filters
    const auto input_channels = std::max(
            static_cast<size_t>(output_channels),
            static_cast<size_t>(get_input_channels()));
    auto conv_out = std::vector<float>(output_channels * width * height);

    auto V = std::vector<float>(WINOGRAD_TILE * input_channels * tiles);
    auto M = std::vector<float>(WINOGRAD_TILE * output_channels * tiles);

    std::vector<float> policy_data(Network::NUM_POLICY_INPUT_PLANES * width * height);
    std::vector<float> value_data(Network::NUM_VALUE_INPUT_PLANES * width * height);

    winograd_convolve3(output_channels, input, conv_weights[0], V, M, conv_out);
    batchnorm<64>(output_channels, conv_out,
                  batchnorm_means[0].data(),
                  batchnorm_stddivs[0].data());

    // Residual tower
    auto conv_in = std::vector<float>(output_channels * width * height);
    auto res = std::vector<float>(output_channels * width * height);
    for (auto i = size_t{1}; i < conv_weights.size(); i += 2) {
        auto output_channels = conv_biases[i].size();
        std::swap(conv_out, conv_in);
        std::copy(begin(conv_in), end(conv_in), begin(res));
        winograd_convolve3(output_channels, conv_in,
                           conv_weights[i], V, M, conv_out);
        batchnorm<64>(output_channels, conv_out,
                      batchnorm_means[i].data(),
                      batchnorm_stddivs[i].data());

        output_channels = conv_biases[i + 1].size();
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           conv_weights[i + 1], V, M, conv_out);
        batchnorm<64>(output_channels, conv_out,
                      batchnorm_means[i + 1].data(),
                      batchnorm_stddivs[i + 1].data(),
                      res.data());
    }
    convolve<1>(NUM_POLICY_INPUT_PLANES, conv_out, conv_pol_w, conv_pol_b, policy_data);
    convolve<1>(NUM_VALUE_INPUT_PLANES, conv_out, conv_val_w, conv_val_b, value_data);
    batchnorm<width*height>(NUM_POLICY_INPUT_PLANES, policy_data, bn_pol_w1.data(), bn_pol_w2.data());

    batchnorm<width*height>(NUM_VALUE_INPUT_PLANES, value_data, bn_val_w1.data(), bn_val_w2.data());

    if (m_format_version == 1) {
        innerproduct<NUM_POLICY_INPUT_PLANES*width*height, V1_NUM_OUTPUT_POLICY>(policy_data, v1_ip_pol_w, v1_ip_pol_b, output_pol);
    } else {
        innerproduct<NUM_POLICY_INPUT_PLANES*width*height, V2_NUM_OUTPUT_POLICY>(policy_data, v2_ip_pol_w, v2_ip_pol_b, output_pol);
    }
    innerproduct<NUM_VALUE_INPUT_PLANES*width*height, NUM_VALUE_CHANNELS>(value_data, ip1_val_w, ip1_val_b, output_val);
}

template<typename T>
T relative_difference(T a, T b) {
    // Handle NaN
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<T>::max();
    }

    constexpr auto small_number = 1e-3f;
    auto fa = std::fabs(a);
    auto fb = std::fabs(b);

    if (fa > small_number && fb > small_number) {
        // Handle sign difference
        if (((a < 0) != (b < 0)) && (a != 0) && (b != 0)) {
            return std::numeric_limits<T>::max();
        }
    }

    // Handle underflow
    fa = std::max(fa, small_number);
    fb = std::max(fb, small_number);

    return std::max(fabs((fa - fb) / fa), fabs((fa - fb) / fb));
}

bool compare_net_outputs(std::vector<float>& data,
                         std::vector<float>& ref,
                         bool& fatal,
                         bool display_only = false,
                         std::string info = "") {
    auto almost_equal = true;
    // The idea is to allow an OpenCL error > 10% every SELFCHECK_MIN_EXPANSIONS
    // correct expansions. As the num_expansions increases between errors > 10%,
    // we'll allow more errors to occur (max 3) before crashing. As if it
    // builds up credit.
    constexpr int64 min_correct_expansions = SELFCHECK_MIN_EXPANSIONS / SELFCHECK_PROBABILITY / 2;
    static_assert(min_correct_expansions > 0, "Increase minimal nof expansions");
    static std::atomic<int64> num_expansions{min_correct_expansions};
    num_expansions = std::min(num_expansions + 1, 3 * min_correct_expansions);

    // We accept an error up to 10%, but output values
    // smaller than 1/1000th are "rounded up" for the comparison.
    constexpr float relative_error = 10e-2f;
    for (auto idx = size_t{0}; idx < data.size(); ++idx) {
        auto err = relative_difference(data[idx], ref[idx]);
        if (display_only) {
            myprintf("compare_net_outputs %s idx %d data %f ref %f err=%f\n",
                info.c_str(), idx, data[idx], ref[idx], err);
        } else if (err > relative_error) {
            almost_equal = false;
            myprintf("Error in OpenCL calculation: expected %f got %f (%lli"
                       "(error=%f%%)\n", ref[idx], data[idx], num_expansions.load(), err * 100.0);
            if (num_expansions < min_correct_expansions) {
                fatal = true;
            }
            else {
                num_expansions -= min_correct_expansions;
            }
        }
    }
    return almost_equal;
}
#endif

void Network::softmax(const std::vector<float>& input,
                      std::vector<float>& output,
                      float temperature) {
    assert(&input != &output);

    auto alpha = *std::max_element(begin(input),
                                   begin(input) + output.size());
    alpha /= temperature;

    auto denom = 0.0f;
    auto helper = std::vector<float>(output.size());
    for (auto i = size_t{0}; i < output.size(); i++) {
        auto val   = std::exp((input[i]/temperature) - alpha);
        helper[i]  = val;
        denom     += val;
    }
    for (auto i = size_t{0}; i < output.size(); i++) {
        output[i] = helper[i] / denom;
    }
}

Network::Netresult Network::get_scored_moves(const BoardHistory& pos, DebugRawData* debug_data, bool skip_cache) {
    Netresult result;
    auto full_key = pos.cur().full_key();
    // Mix in the history to try and ensure no false positive lookups where
    // the cached values would differ from the NN evals.
    int history_count = pos.positions.size();
    for (int i = history_count - 2; i >= std::max(0, history_count - T_HISTORY); i--) {
        full_key *= 31;
        full_key ^= pos.positions[i].full_key();
    }

    // See if we already have this in the cache.
    if (!skip_cache) {
        if (NNCache::get_NNCache().lookup(full_key, result)) {
            return result;
        }
    }

    NNPlanes planes;
    gather_features(pos, planes);
    result = get_scored_moves_internal(pos, planes, debug_data);

    // Insert result into cache.
    NNCache::get_NNCache().insert(full_key, result);

    return result;
}

Network::Netresult Network::get_scored_moves_internal(const BoardHistory& pos, NNPlanes& planes, DebugRawData* debug_data) {
    // NNPlanes is sized to support either version, so this assert uses
    // MAX_INPUT_CHANNELS. The rest of the code uses get_input_channels()
    // to match the actual number of bits expected by each network.
    assert(MAX_INPUT_CHANNELS == planes.bit.size()+3);
    constexpr int width = 8;
    constexpr int height = 8;
    const auto convolve_channels = conv_pol_w.size() / conv_pol_b.size();
    std::vector<net_t> input_data;
    std::vector<net_t> output_data(convolve_channels * width * height);
    std::vector<float> value_data(Network::NUM_VALUE_INPUT_PLANES * width * height);
    std::vector<float> policy_data(get_num_output_policy());
    std::vector<float> softmax_data(get_num_output_policy());
    std::vector<float> winrate_data(Network::NUM_VALUE_CHANNELS);
    std::vector<float> winrate_out(1);
    // Data layout is input_data[(c * height + h) * width + w]
    input_data.reserve(get_input_channels() * width * height);
    for (int c = 0; c < get_input_channels() - 3; ++c) {
        for (int i = 0; i < 64; ++i) {
            input_data.emplace_back(net_t(planes.bit[c][i]));
        }
    }
    for (int i = 0; i < 64; ++i) {
        input_data.emplace_back(net_t(planes.rule50_count));
    }
    for (int i = 0; i < 64; ++i) {
        input_data.emplace_back(net_t(planes.move_count));
    }
    // TODO: I changed this to a plane of ones for V2.
    // To help see the edge of the board
    for (int i = 0; i < 64; ++i) {
        input_data.emplace_back(net_t(m_format_version == 1 ? 0.0 : 1.0));
    }
    assert(input_data.size() == get_input_channels() * width * height);
#ifdef USE_OPENCL
    opencl.forward(input_data, policy_data, value_data);
#elif defined(USE_BLAS) && !defined(USE_OPENCL)
    forward_cpu(input_data, policy_data, value_data);
#endif
#ifdef USE_OPENCL_SELFCHECK
    // Both implementations are available, self-check the OpenCL driver by
    // running both with a probability of 1/2000.
    if (Random::GetRng().RandInt(SELFCHECK_PROBABILITY) == 0) {
        auto cpu_policy_data = std::vector<float>(policy_data.size());
        auto cpu_value_data = std::vector<float>(value_data.size());
        auto fatal = false;
        forward_cpu(input_data, cpu_policy_data, cpu_value_data);
        auto almost_equal = compare_net_outputs(policy_data, cpu_policy_data, fatal);
        almost_equal &= compare_net_outputs(value_data, cpu_value_data, fatal);
        if (!almost_equal) {
            myprintf("PGN\n%s\nEND\n", pos.pgn().c_str());
            // Compare again but with debug info
            compare_net_outputs(policy_data, cpu_policy_data, fatal, true, "orig policy");
            compare_net_outputs(value_data, cpu_value_data, fatal, true, "orig value");
            // Call opencl.forward again to see if the error is reproduceable.
            std::vector<float> value_data_retry(Network::NUM_VALUE_INPUT_PLANES * width * height);
            std::vector<float> policy_data_retry(get_num_output_policy());
            opencl.forward(input_data, policy_data_retry, value_data_retry);
            auto almost_equal_retry = compare_net_outputs(policy_data_retry, policy_data, fatal, true, "retry policy");
            almost_equal_retry &= compare_net_outputs(value_data_retry, value_data, fatal, true, "retry value");
            if (!almost_equal_retry) {
                throw std::runtime_error("OpenCL retry self-check mismatch.");
            } else {
                myprintf("compare_net_outputs retry was ok\n");
            }
            if (fatal) {
                myprintf_so("Update your GPU drivers or reduce the amount of games "
                           "played simultaneously.\n");
                throw std::runtime_error("OpenCL self-check mismatch.");
            }
        }
    }
#endif

    // Get the moves
    softmax(policy_data, softmax_data, cfg_softmax_temp);
    std::vector<float>& outputs = softmax_data;

    // Now get the score
    innerproduct<NUM_VALUE_CHANNELS, 1>(value_data, ip2_val_w, ip2_val_b, winrate_out);

    // Sigmoid
    auto winrate_sig = (1.0f + std::tanh(winrate_out[0])) / 2.0f;

    MoveList<LEGAL> moves(pos.cur());
    std::vector<scored_node> result;
    for (Move move : moves) {
        result.emplace_back(outputs[lookup(move, pos.cur().side_to_move())], move);
    }

    if (debug_data) {
      debug_data->input = input_data;
      debug_data->policy_output = outputs;
      debug_data->value_output = winrate_sig;
      debug_data->filtered_output = result;
    }

    return std::make_pair(result, winrate_sig);
}

template<PieceType Pt>
void addPieces(const Position* pos, Color side, Network::NNPlanes& planes, int plane_idx, bool flip) {
  const Square* squares = pos->squares<Pt>(side);
  while (*squares != SQ_NONE) {
    Square s = *squares++;
    planes.bit[plane_idx][flip ? ~s : s] = true;
  }
}

void Network::gather_features(const BoardHistory& bh, NNPlanes& planes) {
    Color us = bh.cur().side_to_move();
    Color them = ~us;
    const Position* pos = &bh.cur();

    int kFeatureBase = T_HISTORY * get_hist_planes();
    if (pos->can_castle(BLACK_OOO)) planes.bit[kFeatureBase+(us==BLACK?0:2)+0].set();
    if (pos->can_castle(BLACK_OO)) planes.bit[kFeatureBase+(us==BLACK?0:2)+1].set();
    if (pos->can_castle(WHITE_OOO)) planes.bit[kFeatureBase+(us==WHITE?0:2)+0].set();
    if (pos->can_castle(WHITE_OO)) planes.bit[kFeatureBase+(us==WHITE?0:2)+1].set();
    if (us == BLACK) planes.bit[kFeatureBase+4].set();
    planes.rule50_count = pos->rule50_count();
    // Move count is redundant in chess and was clamped to uint8_t. We disabled
    // in training and so we should disable the move_count plane as input to
    // the NN here.
    planes.move_count = 0;

    int mc = bh.positions.size() - 1;
    bool flip = us == BLACK;
    for (int i = 0; i < std::min(T_HISTORY, mc + 1); ++i) {
        pos = &bh.positions[mc - i];

        if (m_format_version == 1) {
            us = pos->side_to_move();
            them = ~us;

            flip = us == BLACK;
        }

        addPieces<PAWN  >(pos, us, planes, i * get_hist_planes() + 0, flip);
        addPieces<KNIGHT>(pos, us, planes, i * get_hist_planes() + 1, flip);
        addPieces<BISHOP>(pos, us, planes, i * get_hist_planes() + 2, flip);
        addPieces<ROOK  >(pos, us, planes, i * get_hist_planes() + 3, flip);
        addPieces<QUEEN >(pos, us, planes, i * get_hist_planes() + 4, flip);
        addPieces<KING  >(pos, us, planes, i * get_hist_planes() + 5, flip);

        addPieces<PAWN  >(pos, them, planes, i * get_hist_planes() + 6, flip);
        addPieces<KNIGHT>(pos, them, planes, i * get_hist_planes() + 7, flip);
        addPieces<BISHOP>(pos, them, planes, i * get_hist_planes() + 8, flip);
        addPieces<ROOK  >(pos, them, planes, i * get_hist_planes() + 9, flip);
        addPieces<QUEEN >(pos, them, planes, i * get_hist_planes() + 10, flip);
        addPieces<KING  >(pos, them, planes, i * get_hist_planes() + 11, flip);

        int repetitions = pos->repetitions_count();
        if (repetitions >= 1) planes.bit[i * get_hist_planes() + 12].set();
        if (m_format_version == 1) {
            if (repetitions >= 2) planes.bit[i * get_hist_planes() + 13].set();
        }
    }
}

std::string Network::DebugRawData::getJson() const {
  std::stringstream s;
  s << "{\n\"value_output\":" << value_output << ",\n";
  s << "\"input\":[";
  for (size_t i = 0; i < input.size(); ++i) {
    if (i != 0) s << ",";
    s << input[i];
  }
  s << "],\n";
  s << "\"policy_output\":[";
  for (size_t i = 0; i < policy_output.size(); ++i) {
    if (i != 0) s << ",";
    s << policy_output[i];
  }
  s << "],\n";
  s << "\"filtered_output\":[";
  for (size_t i = 0; i < filtered_output.size(); ++i) {
    if (i != 0) s << ",";
    s << "{\"m\":" << filtered_output[i].second << ",\"v\":" << filtered_output[i].first << "}";
  }
  s << "]}\n";
  return s.str();
}
