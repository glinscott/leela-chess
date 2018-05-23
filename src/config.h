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

#ifndef CONFIG_INCLUDED
#define CONFIG_INCLUDED

#ifdef _WIN32
#undef HAVE_SELECT
#define NOMINMAX
#else
#define HAVE_SELECT
#endif

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

#define PROGRAM_VERSION "v0.10"

// OpenBLAS limitation
#if defined(USE_BLAS) && defined(USE_OPENBLAS)
#define MAX_CPUS 64
#else
#define MAX_CPUS 128
#endif

/* Integer types */

typedef int int32;
typedef short int16;
typedef signed char int8;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

/* Data type definitions */

#ifdef _WIN32
typedef __int64 int64;
typedef unsigned __int64 uint64;
#define htole64(x) (x)
#define htole32(x) (x)
#elif defined(__APPLE__)
typedef long long int int64;
typedef  unsigned long long int uint64;
#include <libkern/OSByteOrder.h>
#define htole32(x) OSSwapHostToLittleInt32(x)
#define htole64(x) OSSwapHostToLittleInt64(x)
#else
typedef long long int int64;
typedef  unsigned long long int uint64;
#include <endian.h>
#endif

using net_t = float;

#if (_MSC_VER >= 1400) /* VC8+ Disable all deprecation warnings */
    #pragma warning(disable : 4996)
#endif /* VC8+ */
#endif

