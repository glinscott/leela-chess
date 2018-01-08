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

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <string>
#include <sstream>
#include <cmath>
#include <climits>
#include <algorithm>
#include <random>
#include <chrono>

#include "config.h"
#include "Utils.h"
#include "UCTSearch.h"
#include "UCTNode.h"
#include "Network.h"
#include "TTable.h"
#include "Parameters.h"
#include "Training.h"

using namespace Utils;

// Configuration flags
bool cfg_allow_pondering;
int cfg_num_threads;
int cfg_max_playouts;
int cfg_lagbuffer_cs;
int cfg_resignpct;
int cfg_noise;
int cfg_random_cnt;
int cfg_min_resign_moves;
uint64_t cfg_rng_seed;
bool cfg_dumbpass;
#ifdef USE_OPENCL
std::vector<int> cfg_gpus;
int cfg_rowtiles;
#endif
float cfg_puct;
float cfg_softmax_temp;
std::string cfg_weightsfile;
std::string cfg_logfile;
FILE* cfg_logfile_handle;
bool cfg_quiet;

void Parameters::setup_default_parameters() {
    cfg_allow_pondering = true;
    cfg_num_threads = std::max(1, std::min(SMP::get_num_cpus(), MAX_CPUS));
    //cfg_num_threads = 1;

    cfg_max_playouts = 800;
    cfg_lagbuffer_cs = 100;
#ifdef USE_OPENCL
    cfg_gpus = { };
    cfg_rowtiles = 5;
#endif
    cfg_puct = 0.85f;
    cfg_softmax_temp = 1.0f;
    cfg_min_resign_moves = 20;
    cfg_resignpct = 10;
    cfg_noise = false;
    cfg_random_cnt = 0;
    cfg_dumbpass = false;
    cfg_logfile_handle = nullptr;
    cfg_quiet = false;
    
    // C++11 doesn't guarantee *anything* about how random this is,
    // and in MinGW it isn't random at all. But we can mix it in, which
    // helps when it *is* high quality (Linux, MSVC).
    std::random_device rd;
    std::ranlux48 gen(rd());
    uint64 seed1 = (gen() << 16) ^ gen();
    // If the above fails, this is one of our best, portable, bets.
    uint64 seed2 = std::chrono::high_resolution_clock::
    now().time_since_epoch().count();
    cfg_rng_seed = seed1 ^ seed2;

    cfg_weightsfile = "weights.txt";
}

