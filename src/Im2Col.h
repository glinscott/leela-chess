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

#ifndef IM2COL_H_INCLUDED
#define IM2COL_H_INCLUDED

#include <cassert>
#include <vector>
#include <algorithm>

template <unsigned long filter_size>
void im2col(const size_t channels, const std::vector<net_t>& input, std::vector<float>& output) {
    constexpr size_t height = 8;
    constexpr size_t width = 8;
    constexpr size_t channel_size = height * width;

    constexpr size_t pad = (filter_size / 2);
    constexpr size_t output_h = height + 2 * pad - filter_size  + 1;
    constexpr size_t output_w = width + 2 * pad - filter_size + 1;

    const net_t* data_im = input.data();
    float* data_col = output.data();

    for (size_t channel = channels; channel--; data_im += channel_size) {
        for (size_t kernel_row = 0; kernel_row < filter_size; kernel_row++) {
            for (size_t kernel_col = 0; kernel_col < filter_size; kernel_col++) {
                int input_row = -pad + kernel_row;
                for (size_t output_rows = output_h; output_rows; output_rows--) {
                    if ((unsigned)input_row < height) {
                        int input_col = -pad + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if ((unsigned)input_col < width) {
                                *(data_col++) =
                                    data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col++;
                        }
                    } else {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    }
                    input_row++;
                }
            }
        }
    }
}

template <>
void im2col<1>(const size_t channels,
               const std::vector<net_t>& input,
               std::vector<float>& output) {
    constexpr size_t boardsize = 8;
    auto outSize = size_t{channels * boardsize * boardsize};
    assert(output.size() == outSize);
    std::copy(begin(input), begin(input) + outSize, begin(output));
}

#endif
