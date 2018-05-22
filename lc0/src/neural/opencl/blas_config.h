/*
    This file is part of Leela Zero.

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

// I admit I'm rather uncertain where this particular include belongs in lc0...

#pragma once

#include <cstddef>

/* Features */
#define USE_BLAS
#if !defined(__APPLE__) && !defined(__MACOSX)
#define USE_OPENBLAS
#endif
//#define USE_MKL
#ifndef FEATURE_USE_CPU_ONLY
#define USE_OPENCL
#define USE_OPENCL_SELFCHECK
#endif
static constexpr int SELFCHECK_PROBABILITY = 2000;
static constexpr int SELFCHECK_MIN_EXPANSIONS = 2'000'000;
#define USE_TUNER

// OpenBLAS limitation
#if defined(USE_BLAS) && defined(USE_OPENBLAS)
#define MAX_CPUS 64
#else
#define MAX_CPUS 128
#endif

using net_t = float;

#if (_MSC_VER >= 1400) /* VC8+ Disable all deprecation warnings */
    #pragma warning(disable : 4996)
#endif /* VC8+ */

size_t ceilMultiple(size_t a, size_t b);

static constexpr auto WINOGRAD_ALPHA = 4;
static constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
static constexpr auto WINOGRAD_P = 8 * 8 / 4;
