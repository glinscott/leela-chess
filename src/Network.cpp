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

using namespace Utils;

// Input + residual block tower
static std::vector<std::vector<float>> conv_weights;
static std::vector<std::vector<float>> conv_biases;
static std::vector<std::vector<float>> batchnorm_means;
static std::vector<std::vector<float>> batchnorm_variances;

// Policy head
static std::vector<float> conv_pol_w;
static std::vector<float> conv_pol_b;
static std::array<float, 2> bn_pol_w1;
static std::array<float, 2> bn_pol_w2;

static std::array<float, 261364> ip_pol_w;
static std::array<float, 362> ip_pol_b;

// Value head
static std::vector<float> conv_val_w;
static std::vector<float> conv_val_b;
static std::array<float, 1> bn_val_w1;
static std::array<float, 1> bn_val_w2;

static std::array<float, 92416> ip1_val_w;
static std::array<float, 256> ip1_val_b;

static std::array<float, 256> ip2_val_w;
static std::array<float, 1> ip2_val_b;

//void Network::benchmark(Position* pos, int iterations) //--temporarily (?) killed.

void Network::init() {
#ifdef USE_OPENCL
    myprintf("Initializing OpenCL\n");
    opencl.initialize();

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
    myprintf("%d blocks\nTransferring weights to GPU...", residual_blocks);

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
                batchnorm_variances.emplace_back(weights);
            }
        } else if (linecount == plain_conv_wts) {
            conv_pol_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 1) {
            conv_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 2) {
            std::copy(begin(weights), end(weights), begin(bn_pol_w1));
        } else if (linecount == plain_conv_wts + 3) {
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

    // input
    size_t weight_index = 0;
    opencl_net.push_convolve(3, conv_weights[weight_index],
                                conv_biases[weight_index]);
    opencl_net.push_batchnorm(361, batchnorm_means[weight_index],
                                   batchnorm_variances[weight_index]);
    weight_index++;

    // residual blocks
    for (auto i = size_t{0}; i < residual_blocks; i++) {
        opencl_net.push_residual(3, conv_weights[weight_index],
                                    conv_biases[weight_index],
                                    batchnorm_means[weight_index],
                                    batchnorm_variances[weight_index],
                                    conv_weights[weight_index + 1],
                                    conv_biases[weight_index + 1],
                                    batchnorm_means[weight_index + 1],
                                    batchnorm_variances[weight_index + 1]);
        weight_index += 2;
    }
    myprintf("done\n");
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

#ifdef USE_BLAS
template<unsigned int filter_size,
         unsigned int outputs>
void convolve(const std::vector<net_t>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    // fixed for 19x19
    constexpr unsigned int width = 19;
    constexpr unsigned int height = 19;
    constexpr unsigned int board_squares = width * height;
    constexpr unsigned int filter_len = filter_size * filter_size;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * board_squares == output.size());

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 22 3 3
    // outputs[96,19x19] = weights[96,22x3x3] x col[22x3x3,19x19]
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

template<unsigned int channels, unsigned int spatial_size>
void batchnorm(const std::vector<float>& input, const std::array<float, channels>& means, const std::array<float, channels>& variances, std::vector<float>& output)
{
    constexpr float epsilon = 1e-5f;

    auto lambda_ReLU = [](float val) { return (val > 0.0f) ? val : 0.0f; };

    for (unsigned int c = 0; c < channels; ++c) {
        float mean = means[c];
        float variance = variances[c] + epsilon;
        float scale_stddiv = 1.0f / std::sqrt(variance);

        float * out = &output[c * spatial_size];
        float const * in  = &input[c * spatial_size];
        for (unsigned int b = 0; b < spatial_size; b++) {
            out[b] = lambda_ReLU(scale_stddiv * (in[b] - mean));
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

Network::Netresult Network::get_scored_moves(Position* pos) {
    Netresult result;
    NNPlanes planes;
    gather_features(pos, planes);

	result = get_scored_moves_internal(pos, planes);

    return result;
}

Network::Netresult Network::get_scored_moves_internal(Position* pos, NNPlanes& planes) { //--must fix this!
    assert(INPUT_CHANNELS == planes.size());
    constexpr int width = 19;
    constexpr int height = 19;
    const auto convolve_channels = conv_pol_w.size() / conv_pol_b.size();
    std::vector<net_t> input_data;
    std::vector<net_t> output_data(convolve_channels * width * height);
    std::vector<float> policy_data_1(2 * width * height);
    std::vector<float> policy_data_2(2 * width * height);
    std::vector<float> value_data_1(1 * width * height);
    std::vector<float> value_data_2(1 * width * height);
    std::vector<float> policy_out((width * height) + 1);
    std::vector<float> softmax_data((width * height) + 1);
    std::vector<float> winrate_data(256);
    std::vector<float> winrate_out(1);
    // Data layout is input_data[(c * height + h) * width + w]
    input_data.reserve(INPUT_CHANNELS * width * height);
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                input_data.emplace_back(net_t(planes[c][h*8 + w]));
            }
        }
    }
#ifdef USE_OPENCL
    opencl_net.forward(input_data, output_data);
    // Get the moves
    convolve<1, 2>(output_data, conv_pol_w, conv_pol_b, policy_data_1);
    batchnorm<2, 361>(policy_data_1, bn_pol_w1, bn_pol_w2, policy_data_2);
    innerproduct<2*361, 362>(policy_data_2, ip_pol_w, ip_pol_b, policy_out);
    softmax(policy_out, softmax_data, cfg_softmax_temp);
    std::vector<float>& outputs = softmax_data;

    // Now get the score
    convolve<1, 1>(output_data, conv_val_w, conv_val_b, value_data_1);
    batchnorm<1, 361>(value_data_1, bn_val_w1, bn_val_w2, value_data_2);
    innerproduct<361, 256>(value_data_2, ip1_val_w, ip1_val_b, winrate_data);
    innerproduct<256, 1>(winrate_data, ip2_val_w, ip2_val_b, winrate_out);

    // Sigmoid
    float winrate_sig = (1.0f + std::tanh(winrate_out[0])) / 2.0f;
#elif defined(USE_BLAS) && !defined(USE_OPENCL)
#error "Not implemented"
    // Not implemented yet - not very useful unless you have some
    // sort of Xeon Phi
    softmax(output_data, softmax_data, cfg_softmax_temp);
    // Move scores
    std::vector<float>& outputs = softmax_data;
#endif
    std::vector<scored_node> result;
    for (size_t idx = 0; idx < outputs.size(); idx++) {
        if (idx < 19*19) {
            auto val = outputs[idx];
            auto rot_idx = rotate_nn_idx(idx, rotation);
            int x = rot_idx % 19;
            int y = rot_idx / 19;
            int rot_vtx = state->board.get_vertex(x, y);
            if (state->board.get_square(rot_vtx) == FastBoard::EMPTY) {
                result.emplace_back(val, rot_vtx);
            }
        } else {
            result.emplace_back(outputs[idx], FastBoard::PASS);
        }
    }

    return std::make_pair(result, winrate_sig);
}

//void Network::show_heatmap(Position* state, Netresult& result, bool topmoves) { //--killed.

void Network::gather_features(Position* pos, NNPlanes& planes) {
    planes.resize(INPUT_CHANNELS);
	Color side = pos->side_to_move();
	//--ignoring halfmove clock feature.
	if (pos->can_castle(BLACK_OOO)) planes[0].set();
	if (pos->can_castle(BLACK_OO)) planes[1].set();
	if (pos->can_castle(WHITE_OOO)) planes[2].set();
	if (pos->can_castle(WHITE_OO)) planes[3].set();
	//--ignoring fullmove number feature.
	if (side == BLACK) planes[4].set();
	
	std::stack<StateInfo*> states;
	int backtracks;
	for (backtracks = 0; backtracks < T_HISTORY; backtracks++) {
		const Square* squares;  //--tried doing the following by iterating over enums, but pos->squares didn't like being specialized by a static_cast.
		squares = pos->squares<PAWN>(side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 1][(int) squares[i]] = true;
		}
		squares = pos->squares<KNIGHT>(side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 2][(int) squares[i]] = true;
		}
		squares = pos->squares<BISHOP>(side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 3][(int) squares[i]] = true;
		}
		squares = pos->squares<ROOK>(side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 4][(int) squares[i]] = true;
		}
		squares = pos->squares<QUEEN>(side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 5][(int) squares[i]] = true;
		}
		squares = pos->squares<KING>(side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 6][(int) squares[i]] = true;
		}
		Color not_side = side == WHITE ? BLACK : WHITE;
		squares = pos->squares<PAWN>(not_side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 7][(int) squares[i]] = true;
		}
		squares = pos->squares<KNIGHT>(not_side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 8][(int) squares[i]] = true;
		}
		squares = pos->squares<BISHOP>(not_side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 9][(int) squares[i]] = true;
		}
		squares = pos->squares<ROOK>(not_side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 10][(int) squares[i]] = true;
		}
		squares = pos->squares<QUEEN>(not_side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 11][(int) squares[i]] = true;
		}
		squares = pos->squares<KING>(not_side);
		for (int i = 0; i < 16; i++) {
			if (squares[i] == SQ_NONE) break;
			planes[INPUT_CHANNELS - backtracks * 7 - 12][(int) squares[i]] = true;
		}
		int repetitions = pos->repetitions_count();
		if (repetitions >= 3) planes[INPUT_CHANNELS - backtracks * 7 - 13].set();
		if (repetitions >= 2) planes[INPUT_CHANNELS - backtracks * 7 - 14].set();
		
		StateInfo* state = pos->get_state();
		if (state->move == MOVE_NONE) break;
		states.push(state);
		pos->undo_move(state->move);
	}
	
	for (int h = 0; h < backtracks; h++) {
		StateInfo* state = states.top();
		states.pop();
		pos->do_move(state->move, *state); //this should leave *state just as it was before... :(
	}
}
