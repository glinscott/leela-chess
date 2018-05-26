/*
    This file is part of Leela Chess Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto
    Copyright (C) 2018 The LCZero Authors

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

#include "blas_transforms.h"
#include <cblas.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <array>

namespace lczero {

void BlasTransforms::initOneBlock(Weights::ConvBlock& block, bool inputlayer, bool headlayer) {

  if (!headlayer) {
    size_t channels;
    if (inputlayer)
      channels = kInputPlanes;
    else
      channels = block.biases.size();
    block.weights = winograd_transform_f(block.weights, block.biases.size(), channels);
  }

  // the weights stored are actually variances, not stddivs, and also all the
  // code downstream assumes that they're inverse stddivs for efficiency
  constexpr float epsilon = 1e-5f;
  for (auto& weight : block.bn_stddivs) {
    weight = 1.0f / std::sqrt(weight + epsilon);
  }

  // Biases are not calculated and are typically zero but some networks might
  // still have non-zero biases.
  // Move biases to batchnorm means to make the output match without having
  // to separately add the biases.
  for (auto j = size_t{0}; j < block.bn_means.size(); j++) {
    block.bn_means[j] -= block.biases[j];
    block.biases[j] = 0.0f;
  }
}

std::vector<float> BlasTransforms::winograd_transform_f(const std::vector<float>& f, const int outputs, const int channels) {
  // F(2x2, 3x3) Winograd filter transformation
  // transpose(G.dot(f).dot(G.transpose()))
  // U matrix is transposed for better memory layout in SGEMM
  auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
  auto G = std::array<float, WINOGRAD_TILE>{1.0,  0.0,  0.0,
                                            0.5,  0.5,  0.5,
                                            0.5, -0.5,  0.5,
                                            0.0,  0.0,  1.0};
  auto temp = std::array<float, 12>{};

  for (auto o = 0; o < outputs; o++) {
    for (auto c = 0; c < channels; c++) {
      for (auto i = 0; i < 4; i++) {
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

void BlasTransforms::innerproduct(const std::vector<float>& inputs,
                               const std::vector<float>& weights,
                               const std::vector<float>& biases,
                               std::vector<float>& outputs,
                               bool apply_relu) {
  assert(outputs.size() == biases.size());

  cblas_sgemv(CblasRowMajor, CblasNoTrans,
              // M           K
              outputs.size(), inputs.size(),
              1.0f, weights.data(), inputs.size(),
              inputs.data(), 1, 0.0f,
              outputs.data(), 1);

  auto lambda_ReLU = [](float val) { return (val > 0.0f) ? val : 0.0f; };

  for (size_t o = 0; o < outputs.size(); o++) {
    float val = biases[o] + outputs[o];
    if (apply_relu) { // TODO: for value head fully connected layer in blas-cpu
      val = lambda_ReLU(val);
    }
    outputs[o] = val;
  }
}

/****************************************************************************/
// Now the CPU-only functions. This is pretty much all just copy and pasted
// from lczero/Network.cpp. Can be quirky code.

void BlasTransforms::winograd_transform_in(const std::vector<float>& in,
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

void BlasTransforms::winograd_sgemm(const std::vector<float>& U,
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

void BlasTransforms::winograd_transform_out(const std::vector<float>& M,
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

void BlasTransforms::winograd_convolve3(const int outputs,
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

void BlasTransforms::convolve(size_t outputs,
                           const std::vector<float>& input,
                           const std::vector<float>& weights,
                           const std::vector<float>& biases,
                           std::vector<float>& output) {
    // lczero calls im2col<filter_size>, but only ever with filter_size=1,
    // whose im2col specialization was overridden to be the simple memcpy
    // commented below. I have no idea why the code was like that in the first place.
    constexpr unsigned int filter_len = 1; // = filter_size * filter_size
    // fixed for 8x8
    constexpr unsigned int width = 8;
    constexpr unsigned int height = 8;
    constexpr unsigned int board_squares = width * height;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * board_squares == output.size());

    std::vector<float> col(filter_dim * width * height);
    //im2col<filter_size>(input_channels, input, col);
    // lczero only has filter_size, and filter_size=1 was specialized to merely this memcpy:
    std::copy(begin(input), begin(input)+col.size(), begin(col));
    // TODO: wtf?

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

// again, another lczero function with pointless templating. formerly had
// template <size_t spatial_size>, but only ever called with <64>.
void BlasTransforms::batchnorm(size_t channels,
                            std::vector<float>& data,
                            const std::vector<float>& means,
                            const std::vector<float>& stddivs,
                            const float* eltwise)
{
    constexpr size_t spatial_size = 64; // either literal or width*height in forward_cpu
    auto lambda_ReLU = [](float val) { return (val > 0.0f) ?
                                       val : 0.0f; };

    for (auto c = size_t{0}; c < channels; ++c) {
        auto mean = means[c];
        auto scale_stddiv = stddivs[c];

        // TODO: why all the nonsense with pointers???
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

std::vector<float> BlasTransforms::softmax(const std::vector<float>& inputs, float temperature) {
  auto outputs = std::vector<float>(inputs.size());
  auto alpha = *std::max_element(begin(inputs), end(inputs));
  alpha /= temperature;
  auto denom = 0.0f;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto val = std::exp((inputs[i]/temperature) - alpha);
    outputs[i] = val;
    denom += val;
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i] /= denom;
  }
  return outputs;
}

std::vector<float> BlasTransforms::zeropad_U(const std::vector<float>& U, const int outputs, const int channels, const int outputs_pad, const int channels_pad) {
  // Fill with zeroes
  auto Upad = std::vector<float>(WINOGRAD_TILE * outputs_pad * channels_pad);

  for(auto o = 0; o < outputs; o++) {
      for(auto c = 0; c < channels; c++) {
          for(auto xi = 0; xi < WINOGRAD_ALPHA; xi++){
              for(auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                  Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                     + nu * (outputs_pad * channels_pad)
                     + c * outputs_pad +
                       o]
                  =
                  U  [xi * (WINOGRAD_ALPHA * outputs * channels)
                    + nu * (outputs * channels)
                    + c * outputs
                    + o];
              }
          }
      }
  }
  return Upad;
}

} // namespace lczero
