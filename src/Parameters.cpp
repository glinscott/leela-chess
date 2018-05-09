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
#include "Parameters.h"
#include "Training.h"

using namespace Utils;

// Configuration flags
bool cfg_allow_pondering;
bool cfg_noinitialize;
int cfg_max_threads;
int cfg_num_threads;
int cfg_max_playouts;
int cfg_max_nodes;
int cfg_lagbuffer_ms;
int cfg_resignpct;
int cfg_noise;
int cfg_randomize;
int cfg_timemanage;
int cfg_slowmover;
int cfg_min_resign_moves;
int cfg_root_temp_decay;
// Don't pick any moves with less than this proportion of visits
float cfg_rand_visit_floor;
// Don't pick any moves with eval this much worst than the top move
float cfg_rand_eval_maxdiff;
uint64_t cfg_rng_seed;
#ifdef USE_OPENCL
std::vector<int> cfg_gpus;
bool cfg_sgemm_exhaustive;
bool cfg_tune_only;
#endif
float cfg_puct;
float cfg_softmax_temp;
float cfg_fpu_reduction;
bool cfg_fpu_dynamic_eval;
std::string cfg_weightsfile;
std::string cfg_syzygypath; 
bool cfg_syzygydraw;
std::string cfg_logfile;
std::string cfg_supervise;
FILE* cfg_logfile_handle;
bool cfg_quiet;
bool cfg_go_nodes_as_playouts;

void Parameters::setup_default_parameters() {
    cfg_allow_pondering = false;
    cfg_noinitialize = false;
    int num_cpus = std::thread::hardware_concurrency();
    cfg_max_threads = std::max(1, std::min(num_cpus, MAX_CPUS));
    cfg_num_threads = 2;

    cfg_max_playouts = MAXINT_DIV2;
    cfg_max_nodes    = 800;
    cfg_lagbuffer_ms = 50;
#ifdef USE_OPENCL
    cfg_gpus = { };
    cfg_sgemm_exhaustive = false;
    cfg_tune_only = false;
#endif
    cfg_puct = 0.6f;
    cfg_softmax_temp = 1.0f;
    cfg_fpu_reduction = 0.1f;
    cfg_fpu_dynamic_eval = true;
    cfg_root_temp_decay = 0;
    //cfg_rand_visit_floor = 0.01f;
    cfg_rand_visit_floor = 0.0f;  // Disable for now
    //cfg_rand_eval_maxdiff = 0.05f;
    cfg_rand_eval_maxdiff = 1.0f; // Disable for now by allowing large diffs
    cfg_min_resign_moves = 20;
    cfg_resignpct = -1;
    cfg_noise = false;
    cfg_randomize = false;
    cfg_timemanage = true;
    cfg_slowmover = 89;
    cfg_logfile_handle = nullptr;
    cfg_quiet = false;
    cfg_rng_seed = 0;
    cfg_weightsfile = "weights.txt";
    cfg_syzygypath = "syzygy";
    cfg_syzygydraw = true;
    cfg_go_nodes_as_playouts = false;
}

