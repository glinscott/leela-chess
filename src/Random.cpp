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

#include <thread>

#include "Parameters.h"
#include "Random.h"

Random::Random(std::uint64_t seed) {
  if (seed == 0) {
    std::size_t thread_id =
        std::hash<std::thread::id>()(std::this_thread::get_id());
    seed = cfg_rng_seed ^ (std::uint64_t)thread_id;
  }
  rand_engine_.seed(seed);
}

Random& Random::GetRng(void) {
  static thread_local Random rng{0};
  return rng;
}

std::uint64_t Random::operator()() {
  return RandInt<std::uint64_t>();
}
