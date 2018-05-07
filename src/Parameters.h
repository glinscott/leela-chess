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

#ifndef GTP_H_INCLUDED
#define GTP_H_INCLUDED

#include <string>
#include <vector>

constexpr int MAXINT_DIV2 = std::numeric_limits<int>::max() / 2;
extern bool cfg_allow_pondering;
extern bool cfg_noinitialize;
extern int cfg_max_threads;
extern int cfg_num_threads;
extern int cfg_max_playouts;
extern int cfg_max_nodes;
extern int cfg_lagbuffer_ms;
extern int cfg_resignpct;
extern int cfg_noise;
extern int cfg_randomize;
extern int cfg_timemanage;
extern int cfg_slowmover;
extern int cfg_min_resign_moves;
extern int cfg_root_temp_decay;
extern float cfg_rand_visit_floor;
extern float cfg_rand_eval_maxdiff;
extern uint64_t cfg_rng_seed;
#ifdef USE_OPENCL
extern std::vector<int> cfg_gpus;
extern bool cfg_sgemm_exhaustive;
extern bool cfg_tune_only;
#endif
extern float cfg_puct;
extern float cfg_softmax_temp;
extern float cfg_fpu_reduction;
extern bool cfg_fpu_dynamic_eval;
extern std::string cfg_logfile;
extern std::string cfg_weightsfile;
extern std::string cfg_syzygypath;
extern bool cfg_syzygydraw;
extern std::string cfg_supervise;
extern FILE* cfg_logfile_handle;
extern bool cfg_quiet;
extern bool cfg_go_nodes_as_playouts;

class Parameters {
public:
    static void setup_default_parameters();
private:
    static const std::string s_commands[];
};


#endif

