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

#include <time.h>
#include <limits.h>
#include <thread>
#include "config.h"

#include "Random.h"
#include "Utils.h"
#include "Parameters.h"

Random& Random::get_Rng(void) {
    static thread_local Random s_rng{0};
    return s_rng;
}

Random::Random(uint64 seed) {
    if (seed == 0) {  //--is this necessary? consider reverting to Stockfish-style initialize seed directly, i.e. get rid of this branch.
        size_t thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
        s = cfg_rng_seed ^ (uint64_t)thread_id;
    } else {
        s = seed;
    }
}

uint16 Random::randuint16(const uint16 max) {
    return ((rand64() >> 48) * max) >> 16;
}

uint32 Random::randuint32(const uint32 max) {
    return ((rand64() >> 32) * (uint64)max) >> 32;
}

uint32 Random::randuint32() {
    return rand64() >> 32;
}

float Random::randflt(void) {
    // We need a 23 bit mantissa + implicit 1 bit = 24 bit number
    // starting from a 64 bit random.
    constexpr float umax = 1.0f / (UINT32_C(1) << 24);
    uint32 num = rand64() >> 40;
    return ((float)num) * umax;
}

