/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2017 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <boost/program_options.hpp>
#include <iostream>
#include <random>

#include "Bitboard.h"
#include "Position.h"
#include "Parameters.h"
#include "Utils.h"
#include "UCI.h"
#include "Random.h"
#include "Network.h"
#include "UCTSearch.h"
#include "Training.h"
#include "Movegen.h"

using namespace Utils;

extern const char* StartFEN;

static void license_blurb() {
    printf(
        "LCZero Copyright (C) 2017  Gary Linscott\n"
        "Based on:"
        "Leela Chess Copyright (C) 2017 benediamond\n"
        "Leela Zero Copyright (C) 2017  Gian-Carlo Pascutto\n"
        "Stockfish Copyright (C) 2017  Tord Romstad, Marco Costalba, Joona Kiiski, Gary Linscott\n"
        "This program comes with ABSOLUTELY NO WARRANTY.\n"
        "This is free software, and you are welcome to redistribute it\n"
        "under certain conditions; see the COPYING file for details.\n\n"
    );
}

static void parse_commandline(int argc, char *argv[]) {
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description v_desc("Allowed options");
    v_desc.add_options()
        ("help,h", "Show commandline options.")
        ("threads,t", po::value<int>()->default_value
                      (std::min(2, cfg_num_threads)),
                      "Number of threads to use.")
        ("playouts,p", po::value<int>(),
                       "Weaken engine by limiting the number of playouts. "
                       "Requires --noponder.")
        ("resignpct,r", po::value<int>()->default_value(cfg_resignpct),
                        "Resign when winrate is less than x%.")
        ("noise,n", "Enable policy network randomization.")
        ("seed,s", po::value<std::uint64_t>(),
                   "Random number generation seed.")
        ("weights,w", po::value<std::string>(), "File with network weights.")
        ("logfile,l", po::value<std::string>(), "File to log input/output to.")
        ("quiet,q", "Disable all diagnostic output.")
        ("noponder", "Disable thinking on opponent's time.")
#ifdef USE_OPENCL
        /*
        ("gpu",  po::value<std::vector<int> >(),
                "ID of the OpenCL device(s) to use (disables autodetection).")
        ("rowtiles", po::value<int>()->default_value(cfg_rowtiles),
                     "Split up the board in # tiles.")
        */
#endif
#ifdef USE_TUNER
        ("puct", po::value<float>())
        ("softmax_temp", po::value<float>())
#endif
        ;
    // These won't be shown, we use them to catch incorrect usage of the
    // command line.
    po::options_description h_desc("Hidden options");
    h_desc.add_options()
        ("arguments", po::value<std::vector<std::string>>());
    // Parse both the above, we will check if any of the latter are present.
    po::options_description all("All options");
    all.add(v_desc).add(h_desc);
    po::positional_options_description p_desc;
    p_desc.add("arguments", -1);
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                  .options(all).positional(p_desc).run(), vm);
        po::notify(vm);
    }  catch(const boost::program_options::error& e) {
        myprintf("ERROR: %s\n", e.what());
        license_blurb();
        std::cout << v_desc << std::endl;
        exit(EXIT_FAILURE);
    }

    // Handle commandline options
    if (vm.count("help") || vm.count("arguments")) {
        auto ev = EXIT_SUCCESS;
        // The user specified an argument. We don't accept any, so explain
        // our usage.
        if (vm.count("arguments")) {
            for (auto& arg : vm["arguments"].as<std::vector<std::string>>()) {
                std::cout << "Unrecognized argument: " << arg << std::endl;
            }
            ev = EXIT_FAILURE;
        }
        license_blurb();
        std::cout << v_desc << std::endl;
        exit(ev);
    }

    if (vm.count("quiet")) {
        cfg_quiet = true;
    }

#ifdef USE_TUNER
    if (vm.count("puct")) {
        cfg_puct = vm["puct"].as<float>();
    }
    if (vm.count("softmax_temp")) {
        cfg_softmax_temp = vm["softmax_temp"].as<float>();
    }
#endif

    if (vm.count("logfile")) {
        cfg_logfile = vm["logfile"].as<std::string>();
        myprintf("Logging to %s.\n", cfg_logfile.c_str());
        cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
    }

    if (vm.count("weights")) {
        cfg_weightsfile = vm["weights"].as<std::string>();
    } else {
        myprintf("A network weights file is required to use the program.\n");
        exit(EXIT_FAILURE);
    }

    if (vm.count("threads")) {
        int num_threads = vm["threads"].as<int>();
        if (num_threads > cfg_num_threads) {
            myprintf("Clamping threads to maximum = %d\n", cfg_num_threads);
        } else if (num_threads != cfg_num_threads) {
            myprintf("Using %d thread(s).\n", num_threads);
            cfg_num_threads = num_threads;
        }
    }

    if (vm.count("seed")) {
        cfg_rng_seed = vm["seed"].as<std::uint64_t>();
        if (cfg_num_threads > 1) {
            myprintf("Seed specified but multiple threads enabled.\n");
            myprintf("Games will likely not be reproducible.\n");
        }
    }
    myprintf("RNG seed: %llu\n", cfg_rng_seed);

    if (vm.count("noponder")) {
        cfg_allow_pondering = false;
    }

    if (vm.count("noise")) {
        cfg_noise = true;
    }

    if (vm.count("playouts")) {
        cfg_max_playouts = vm["playouts"].as<int>();
        if (!vm.count("noponder")) {
            myprintf("Nonsensical options: Playouts are restricted but "
                     "thinking on the opponent's time is still allowed. "
                     "Add --noponder if you want a weakened engine.\n");
            exit(EXIT_FAILURE);
        }
    }

    if (vm.count("resignpct")) {
        cfg_resignpct = vm["resignpct"].as<int>();
    }

#ifdef USE_OPENCL
    /*
    if (vm.count("gpu")) {
        cfg_gpus = vm["gpu"].as<std::vector<int> >();
    }

    if (vm.count("rowtiles")) {
        int rowtiles = vm["rowtiles"].as<int>();
        rowtiles = std::min(19, rowtiles);
        rowtiles = std::max(1, rowtiles);
        if (rowtiles != cfg_rowtiles) {
            myprintf("Splitting the board in %d tiles.\n", rowtiles);
            cfg_rowtiles = rowtiles;
        }
    }
    */
#endif
}

void bench() {
  BoardHistory bh;
  bh.positions.emplace_back();
  bh.states.emplace_back(new StateInfo());
  bh.cur().set(StartFEN, bh.states.back().get());

  Network::DebugRawData debug_data;
  auto r = Network::get_scored_moves(bh, &debug_data);

  FILE* f = fopen("/tmp/output", "w");
  fputs(debug_data.getJson().c_str(), f);
  fclose(f);

  /*
  auto search = std::make_unique<UCTSearch>(game, states);
  search->think();
  */
}

// Return the score from the self-play game
int play_one_game(BoardHistory& bh) {
  for (int game_ply = 0; game_ply < 150; ++game_ply) {
    if (bh.cur().is_draw()) {
      return 0;
    }
    MoveList<LEGAL> moves(bh.cur());
    if (moves.size() == 0) {
      if (bh.cur().checkers()) {
        // Checkmate
        return bh.cur().side_to_move() == WHITE ? -1 : 1;
      } else {
        // Stalemate
        return 0;
      }
    }
    auto search = std::make_unique<UCTSearch>(bh.clone());
    Move move = search->think();

    bh.do_move(move);
  }

  // Game termination as draw
  return 0;
}

int play_one_game() {
  BoardHistory bh;
  bh.positions.emplace_back();
  bh.states.emplace_back(new StateInfo());
  bh.cur().set(StartFEN, bh.states.back().get());

  Training::clear_training();
  int game_score = play_one_game(bh);

  printf("%s\n", bh.pgn().c_str());
  printf("Score: %d\n", game_score);

  return game_score;
}

void generate_training_games() {
  auto chunker = OutputChunker{"data/training", true};
  for (;;) {
    Training::dump_training(play_one_game(), chunker);
  }
}

int main(int argc, char* argv[]) {

  Bitboards::init();
  Position::init();

  Parameters::setup_default_parameters();
  parse_commandline(argc, argv);

  // Disable IO buffering as much as possible
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::cin.setf(std::ios::unitbuf);

  setbuf(stdout, nullptr);
  setbuf(stderr, nullptr);
#ifndef WIN32
  setbuf(stdin, nullptr);
#endif
  thread_pool.initialize(cfg_num_threads);
  Network::init();

  // bench();

  // generate_training_games();
  play_one_game();

  return 0;
}
