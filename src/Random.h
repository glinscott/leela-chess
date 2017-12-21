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

#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED

#include "config.h"
#include <cassert>
#include <limits>

/// xorshift64star Pseudo-Random Number Generator
/// This class is based on original code written and dedicated
/// to the public domain by Sebastiano Vigna (2014).
/// It has the following characteristics:
///
///  -  Outputs 64-bit numbers
///  -  Passes Dieharder and SmallCrush test batteries
///  -  Does not require warm-up, no zeroland to escape
///  -  Internal state is a single 64-bit integer
///  -  Period is 2^64 - 1
///  -  Speed: 1.60 ns/call (Core i7 @3.40GHz)
///
/// For further analysis see
///   <http://vigna.di.unimi.it/ftp/papers/xorshift.pdf>


class Random {
public:
    Random() = delete;
    Random(uint64 seed = 0);

    // return the thread local RNG
    static Random& get_Rng(void);

	template<typename T> T rand() { return T(rand64()); }
	
	/// Special generator used to fast init magic numbers.
	/// Output values only have 1/8th of their bits set on average.
	template<typename T> T sparse_rand() { return T(rand64() & rand64() & rand64()); }
	
	uint16 randuint16(const uint16 max);
	uint32 randuint32(const uint32 max);
	uint32 randuint32();
	
	// random float from 0 to 1
	float randflt(void);

	// UniformRandomBitGenerator interface
	using result_type = uint64_t;  //uint64_t instead of uint64...?!
	constexpr static result_type min() {
		return std::numeric_limits<result_type>::min();
	}
	constexpr static result_type max() {
		return std::numeric_limits<result_type>::max();
	}
	result_type operator()() {
		return rand64();
	}

private:
	uint64_t s;
	uint64_t rand64() {
		s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
		return s * 2685821657736338717LL;
	}
};


#endif
