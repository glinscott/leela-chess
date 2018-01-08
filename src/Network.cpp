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

#include "Im2Col.h"
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
#include "OpenCL.h"
#include "UCTNode.h"
#endif

#include "Utils.h"
#include "Random.h"
#include "Network.h"
#include "Utils.h"
#include "Parameters.h"
#include "Timing.h"
#include "Movegen.h"

using namespace Utils;

constexpr int Network::FORMAT_VERSION;
constexpr int Network::T_HISTORY;
constexpr int Network::INPUT_CHANNELS;

constexpr int Network::NUM_OUTPUT_POLICY;
constexpr int Network::NUM_VALUE_CHANNELS;

std::unordered_map<Move, int> Network::move_lookup;

// Input + residual block tower
static std::vector<std::vector<float>> conv_weights;
static std::vector<std::vector<float>> conv_biases;
static std::vector<std::vector<float>> batchnorm_means;
static std::vector<std::vector<float>> batchnorm_stddivs;

// Policy head
static std::vector<float> conv_pol_w;
static std::vector<float> conv_pol_b;
static std::array<float, 2> bn_pol_w1;
static std::array<float, 2> bn_pol_w2;

static std::array<float, Network::NUM_OUTPUT_POLICY*8*8*2> ip_pol_w;
static std::array<float, Network::NUM_OUTPUT_POLICY> ip_pol_b;

// Value head
static std::vector<float> conv_val_w;
static std::vector<float> conv_val_b;
static std::array<float, 1> bn_val_w1;
static std::array<float, 1> bn_val_w2;

static std::array<float, Network::NUM_VALUE_CHANNELS*8*8> ip1_val_w;
static std::array<float, Network::NUM_VALUE_CHANNELS> ip1_val_b;

static std::array<float, Network::NUM_VALUE_CHANNELS> ip2_val_w;
static std::array<float, 1> ip2_val_b;

//void Network::benchmark(Position* pos, int iterations) //--temporarily (?) killed.

void Network::process_bn_var(std::vector<float>& weights, const float epsilon) {
    for (auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

void Network::init() {
    init_move_map();
#ifdef USE_OPENCL
    myprintf("Initializing OpenCL\n");
    opencl.initialize();
#endif

    // Count size of the network
    myprintf("Detecting residual layers...");
    std::ifstream wtfile(cfg_weightsfile);
    if (wtfile.fail()) {
        myprintf("Could not open weights file: %s\n", cfg_weightsfile.c_str());
        exit(EXIT_FAILURE);
    }
    std::string line;
    auto linecount = size_t{0};
    auto format_version = -1;
    while (std::getline(wtfile, line)) {
        std::stringstream iss(line);
        // First line is the file format version id
        if (linecount == 0) {
           iss >> format_version;
           if (iss.fail() || format_version != FORMAT_VERSION) {
               myprintf("Weights file is the wrong version.\n");
               exit(EXIT_FAILURE);
           } else {
               myprintf("v%d...", format_version);
           }
        }
        // Third line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // (Provided they're all equally large - that's not actually required!)
        if (linecount == 2) {
            auto count = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            myprintf("%d channels...", count);
        }
        linecount++;
    }
    // 1 format id, 1 input layer (4 x weights), 14 ending weights,
    // the rest are residuals, every residual has 8 x weight lines
    auto residual_blocks = linecount - (1 + 4 + 14);
    if (residual_blocks % 8 != 0) {
        myprintf("\nInconsistent number of weights in the file.\n");
        exit(EXIT_FAILURE);
    }
    residual_blocks /= 8;
    myprintf("%d blocks\n", residual_blocks);
#ifdef USE_OPENCL
    myprintf("Transferring weights to GPU...");
#endif

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
        float weight;
        std::istringstream iss(line);
        while (iss >> weight) {
            weights.emplace_back(weight);
        }
        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
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
            std::copy(begin(weights), end(weights), begin(ip_pol_w));
        } else if (linecount == plain_conv_wts + 5) {
            std::copy(begin(weights), end(weights), begin(ip_pol_b));
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
    wtfile.close();

#ifdef USE_OPENCL
    // input
    size_t weight_index = 0;
    opencl_net.push_convolve(3, conv_weights[weight_index],
                                conv_biases[weight_index]);
    opencl_net.push_batchnorm(64, batchnorm_means[weight_index],
                                  batchnorm_stddivs[weight_index]);
    weight_index++;

    // residual blocks
    for (auto i = size_t{0}; i < residual_blocks; i++) {
        opencl_net.push_residual(3, conv_weights[weight_index],
                                    conv_biases[weight_index],
                                    batchnorm_means[weight_index],
                                    batchnorm_stddivs[weight_index],
                                    conv_weights[weight_index + 1],
                                    conv_biases[weight_index + 1],
                                    batchnorm_means[weight_index + 1],
                                    batchnorm_stddivs[weight_index + 1]);
        weight_index += 2;
    }
    myprintf("done\n");
#endif
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    // TODO(gary): Figure out a way to do this in a backwards compat way
    // openblas_set_num_threads(1);
    // myprintf("BLAS Core: %s\n", openblas_get_corename());
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
  std::vector<Move> moves;
  for (Square s = SQ_A1; s <= SQ_H8; ++s) {
    // Queen and knight moves
    Bitboard b = attacks_bb(QUEEN, s, 0) | attacks_bb(KNIGHT, s, 0);
    while (b) {
      moves.push_back(make_move(s, pop_lsb(&b)));
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
        moves.push_back(make<PROMOTION>(from, to, QUEEN));
        moves.push_back(make<PROMOTION>(from, to, ROOK));
        moves.push_back(make<PROMOTION>(from, to, BISHOP));
        // Don't need knight, as it's equivalent to pawn push to final rank.
      }
    }
  }

  for (size_t i = 0; i < moves.size(); ++i) {
    move_lookup[moves[i]] = i;
  }
  printf("Generated %lu moves\n", moves.size());
}

#ifdef USE_BLAS
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
        if (outputs == 256) {
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

#ifndef USE_HALF
void Network::forward_cpu(std::vector<float>& input,
                          std::vector<float>& output) {
    // Input convolution
    constexpr int width = 8;
    constexpr int height = 8;
    // Calculate output channels
    const auto output_channels = conv_biases[0].size();
    auto conv_out = std::vector<float>(output_channels * width * height);

    convolve<3>(output_channels, input,
                conv_weights[0], conv_biases[0], conv_out);
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
        convolve<3>(output_channels, conv_in,
                    conv_weights[i], conv_biases[i],
                    conv_out);
        batchnorm<64>(output_channels, conv_out,
                      batchnorm_means[i].data(),
                      batchnorm_stddivs[i].data());

        output_channels = conv_biases[i + 1].size();
        std::swap(conv_out, conv_in);
        convolve<3>(output_channels, conv_in,
                    conv_weights[i + 1], conv_biases[i + 1],
                    conv_out);
        batchnorm<64>(output_channels, conv_out,
                      batchnorm_means[i + 1].data(),
                      batchnorm_stddivs[i + 1].data(),
                      res.data());
    }
    std::copy(begin(conv_out), end(conv_out), begin(output));
}
#endif

template<typename T>
T relative_difference(T a, T b) {
    // Handle NaN
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<T>::max();
    }
    // Handle sign difference
    if (((a < 0) != (b < 0)) && (a != 0) && (b != 0)) {
        return std::numeric_limits<T>::max();
    }
    a = std::fabs(a);
    b = std::fabs(b);

    // Handle underflow
    constexpr float small_number = 1e-3;
    a = std::max(a, small_number);
    b = std::max(b, small_number);

    return std::max(fabs((a - b) / a), fabs((a - b) / b));
}

void compare_net_outputs(std::vector<float>& data,
                         std::vector<float>& ref) {
    // We accept an error up to 5%, but output values
    // smaller than 1/1000th are "rounded up" for the comparison.
    constexpr float relative_error = 5e-2;
    for (auto idx = size_t{0}; idx < data.size(); ++idx) {
        auto err = relative_difference(data[idx], ref[idx]);
        if (err > relative_error) {
            myprintf("Error in OpenCL calculation: expected %f got %f "
                     "(error=%f%%)\n", ref[idx], data[idx], err * 100.0);
            myprintf("Update your GPU drivers or reduce the amount of games "
                     "played simultaneously.\n");
            throw std::runtime_error("OpenCL self-check mismatch.");
        }
    }
}
#endif

void Network::softmax(const std::vector<float>& input, std::vector<float>& output, float temperature) {
    assert(&input != &output);

    float alpha = *std::max_element(input.begin(), input.begin() + output.size());
    alpha /= temperature;

    float denom = 0.0f;
    std::vector<float> helper(output.size());
    for (size_t i = 0; i < output.size(); i++) {
        float val = std::exp((input[i]/temperature) - alpha);
        helper[i] = val;
        denom += val;
    }
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = helper[i] / denom;
    }
}

Network::Netresult Network::get_scored_moves(const BoardHistory& pos, DebugRawData* debug_data) {
    NNPlanes planes;
    gather_features(pos, planes);
    return get_scored_moves_internal(pos, planes, debug_data);
}

Network::Netresult Network::get_scored_moves_internal(const BoardHistory& pos, NNPlanes& planes, DebugRawData* debug_data) {
    assert(INPUT_CHANNELS == planes.bit.size()+3);
    constexpr int width = 8;
    constexpr int height = 8;
    const auto convolve_channels = conv_pol_w.size() / conv_pol_b.size();
    std::vector<net_t> input_data;
    std::vector<net_t> output_data(convolve_channels * width * height);
    std::vector<float> policy_data(2 * width * height);
    std::vector<float> value_data(1 * width * height);
    std::vector<float> policy_out(Network::NUM_OUTPUT_POLICY);
    std::vector<float> softmax_data(Network::NUM_OUTPUT_POLICY);
    std::vector<float> winrate_data(Network::NUM_VALUE_CHANNELS);
    std::vector<float> winrate_out(1);
    // Data layout is input_data[(c * height + h) * width + w]
    input_data.reserve(INPUT_CHANNELS * width * height);
    for (int c = 0; c < INPUT_CHANNELS - 3; ++c) {
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
    for (int i = 0; i < 64; ++i) {
        input_data.emplace_back(net_t(0.0));
    }
    assert(input_data.size() == INPUT_CHANNELS * width * height);
#ifdef USE_OPENCL
    opencl_net.forward(input_data, output_data);
#elif defined(USE_BLAS) && !defined(USE_OPENCL)
    forward_cpu(input_data, output_data);
#endif
#ifdef USE_OPENCL_SELFCHECK
    // Both implementations are available, self-check the OpenCL driver by
    // running both with a probability of 1/2000.
    if (Random::get_Rng().randfix<SELFCHECK_PROBABILITY>() == 0) {
        auto cpu_output_data = std::vector<float>(output_data.size());
        forward_cpu(input_data, cpu_output_data);
        compare_net_outputs(output_data, cpu_output_data);
    }
#endif
    // We calculate both network heads on the CPU. They are irregular
    // and have a much lower compute densitity than the residual layers,
    // which means they don't get much - if any - speedup from being on the
    // GPU. See issue #185.

    // Get the moves
    convolve<1>(2, output_data, conv_pol_w, conv_pol_b, policy_data);
    batchnorm<width*height>(2, policy_data, bn_pol_w1.data(), bn_pol_w2.data());
    innerproduct<2*width*height, Network::NUM_OUTPUT_POLICY>(policy_data, ip_pol_w, ip_pol_b, policy_out);
    softmax(policy_out, softmax_data, cfg_softmax_temp);
    std::vector<float>& outputs = softmax_data;

    // Now get the score
    convolve<1>(1, output_data, conv_val_w, conv_val_b, value_data);
    batchnorm<width*height>(1, value_data, bn_val_w1.data(), bn_val_w2.data());
    innerproduct<width*height, NUM_VALUE_CHANNELS>(value_data, ip1_val_w, ip1_val_b, winrate_data);
    innerproduct<NUM_VALUE_CHANNELS, 1>(winrate_data, ip2_val_w, ip2_val_b, winrate_out);

    // Sigmoid
    float winrate_sig = (1.0f + std::tanh(winrate_out[0])) / 2.0f;

    MoveList<LEGAL> moves(pos.cur());
    std::vector<scored_node> result;
    for (Move move : moves) {
        Move network_move = move;
        if (type_of(move) != PROMOTION || promotion_type(move) == KNIGHT) {
            network_move = Move(network_move & 0xfff);
        }
        // TODO(gary): Remove this once more confident
        if (move_lookup.find(network_move) == move_lookup.end()) {
          printf("Unknown move: %x, %d, %d, %x, %s\n", move, from_sq(move), to_sq(move), type_of(move), pos.cur().move_san(move).c_str());
          exit(1);
        }
        result.emplace_back(outputs[move_lookup.at(network_move)], move);
    }

    if (debug_data) {
      debug_data->input = input_data;
      debug_data->policy_output = outputs;
      debug_data->value_output = winrate_sig;
      debug_data->filtered_output = result;
    }

    return std::make_pair(result, winrate_sig);
}

//void Network::show_heatmap(Position* state, Netresult& result, bool topmoves) { //--killed.

template<PieceType Pt>
void addPieces(const Position* pos, Color side, Network::NNPlanes& planes, int plane_idx) {
  // TODO(gary): Need to flip this to be relative to player to move?
  const Square* squares = pos->squares<Pt>(side);
  while (*squares != SQ_NONE) {
    planes.bit[plane_idx][*squares] = true;
    ++squares;
  }
}

void Network::gather_features(const BoardHistory& bh, NNPlanes& planes) {
    Color side = bh.cur().side_to_move();
    const Position* pos = &bh.cur();

    int kFeatureBase = T_HISTORY * 14;
    if (pos->can_castle(BLACK_OOO)) planes.bit[kFeatureBase+(side==BLACK?0:2)+0].set();
    if (pos->can_castle(BLACK_OO)) planes.bit[kFeatureBase+(side==BLACK?0:2)+1].set();
    if (pos->can_castle(WHITE_OOO)) planes.bit[kFeatureBase+(side==WHITE?0:2)+0].set();
    if (pos->can_castle(WHITE_OO)) planes.bit[kFeatureBase+(side==WHITE?0:2)+1].set();
    if (side == BLACK) planes.bit[kFeatureBase+4].set();
    planes.rule50_count = pos->rule50_count();
    planes.move_count = pos->game_ply();

    int mc = bh.positions.size() - 1;
    for (int i = 0; i < std::min(T_HISTORY, mc + 1); ++i) {
        pos = &bh.positions[mc - i];
        addPieces<PAWN  >(pos, side, planes, i * 14 + 0);
        addPieces<KNIGHT>(pos, side, planes, i * 14 + 1);
        addPieces<BISHOP>(pos, side, planes, i * 14 + 2);
        addPieces<ROOK  >(pos, side, planes, i * 14 + 3);
        addPieces<QUEEN >(pos, side, planes, i * 14 + 4);
        addPieces<KING  >(pos, side, planes, i * 14 + 5);

        addPieces<PAWN  >(pos, ~side, planes, i * 14 + 6);
        addPieces<KNIGHT>(pos, ~side, planes, i * 14 + 7);
        addPieces<BISHOP>(pos, ~side, planes, i * 14 + 8);
        addPieces<ROOK  >(pos, ~side, planes, i * 14 + 9);
        addPieces<QUEEN >(pos, ~side, planes, i * 14 + 10);
        addPieces<KING  >(pos, ~side, planes, i * 14 + 11);

        int repetitions = pos->repetitions_count();
        if (repetitions >= 1) planes.bit[i * 14 + 12].set();
        if (repetitions >= 2) planes.bit[i * 14 + 13].set();
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
