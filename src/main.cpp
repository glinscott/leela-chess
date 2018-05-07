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
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <random>

#include "config.h"
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
#include "pgn.h"
#include "syzygy/tbprobe.h"

using namespace Utils;

static void license_blurb() {
    myprintf_so(
        "LCZero %s Copyright (C) 2017-2018  Gary Linscott and contributors\n"
        "Based on:\n"
        "Leela Chess Copyright (C) 2017 benediamond\n"
        "Leela Zero Copyright (C) 2017-2018  Gian-Carlo Pascutto and contributors\n"
        "Stockfish Copyright (C) 2017  Tord Romstad, Marco Costalba, Joona Kiiski, Gary Linscott\n"
        "This program comes with ABSOLUTELY NO WARRANTY.\n"
        "This is free software, and you are welcome to redistribute it\n"
        "under certain conditions; see the COPYING file for details.\n\n",
        PROGRAM_VERSION
    );
}

static std::string parse_commandline(int argc, char *argv[]) {
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description v_desc("If you have further questions about what an option does, see "
                                   "the project wiki:\n"
                                   "https://github.com/glinscott/leela-chess/wiki\n\n"
                                   "For non-deterministic play that retains strength, use "
                                   "'--noise' or '--tempdecay 10'.\n\n"
                                   "Allowed options");
    v_desc.add_options()
        ("help,h", "Show commandline options.")
        ("threads,t", po::value<int>()->default_value
                      (std::min(cfg_num_threads, cfg_max_threads)),
                      "Number of threads to use.")
        ("playouts,p", po::value<int>(),
                       "Weaken engine by limiting the number of playouts. ")
        ("nodes,v", po::value<int>(),
                       "Weaken engine by limiting the number of nodes in the tree.")
        ("resignpct,r", po::value<int>()->default_value(cfg_resignpct),
                       "Resign when winrate is less than x%.")
        ("noise,n", "Before search begins, add Dirichlet noise to the root node policy's move "
                    "probabilities.")
        ("randomize,m", "After search is complete, select from the moves in proportion to "
                        "their relative values (rather than 'best only').")
        ("tempdecay,d", po::value<int>(),
                       "After search is complete, sometimes pick weaker moves for variety. "
                       "Larger tempdecay values will do this less often, and the effect "
                       "is reduced for moves later in the game. `0` is equivalent to "
                       "--randomize and results in more random moves. This is used by "
                       "self-play games to explore new moves. `10` is a reasonable value, "
                       "and is used by test matches on the server for variety.")
        ("seed,s", po::value<std::uint64_t>(),
                   "Random number generation seed.")
        ("weights,w", po::value<std::string>(), "File with network weights.")
        ("syzygypath,e", po::value<std::string>(), "Folder with syzygy endgame tablebases.")
        ("logfile,l", po::value<std::string>(), "File to log input/output to.")
        ("quiet,q", "Disable all diagnostic output.")
        ("uci", "Don't initialize the engine until \"isready\" command is sent. Use this if your "
                "GUI is freezing on startup.")
        ("start", po::value<std::string>(), "Start command {train, bench}.")
        ("supervise", po::value<std::string>(), "Dump supervised learning data from the pgn.")
#ifdef USE_OPENCL
        ("gpu",  po::value<std::vector<int> >(),
                "ID of the OpenCL device(s) to use (disables autodetection).")
        ("full-tuner", "Try harder to find an optimal OpenCL tuning.")
        ("tune-only", "Tune OpenCL only and then exit.")
#endif
#ifdef USE_TUNER
        ("puct", po::value<float>())
        ("fpu_reduction", po::value<float>())
        ("softmax_temp", po::value<float>())
#endif
        ;
    // These won't be shown, we use them to catch incorrect usage of the
    // command line.
    po::options_description h_desc("Hidden options");
    h_desc.add_options()
        ("arguments", po::value<std::vector<std::string>>())
        ("visits", po::value<int>(),
                     "Weaken engine by limiting the number of visits. This spelling is deprecated.")
        ;
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
    if (vm.count("fpu_reduction")) {
        cfg_fpu_reduction = vm["fpu_reduction"].as<float>();
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

    if (vm.count("supervise")) {
        cfg_supervise = vm["supervise"].as<std::string>();
    }

    if (vm.count("weights")) {
        cfg_weightsfile = vm["weights"].as<std::string>();
    } else if (cfg_supervise.empty()) {
        cfg_weightsfile = "weights.txt";
    }

    if (vm.count("syzygypath")) {
        cfg_syzygypath = vm["syzygypath"].as<std::string>();
    }

    if (vm.count("threads")) {
        int num_threads = vm["threads"].as<int>();
        if (num_threads > cfg_max_threads) {
            myprintf("Clamping threads to maximum = %d\n", cfg_max_threads);
            cfg_num_threads = cfg_max_threads;
        } else {
            myprintf("Using %d thread(s).\n", num_threads);
            cfg_num_threads = num_threads;
        }

    }

    if (vm.count("seed")) {
        cfg_rng_seed = vm["seed"].as<std::uint64_t>();
        if (cfg_rng_seed == 0) {
          myprintf("Nonsensical options: RNG seed cannot be 0.\n");
          exit(EXIT_FAILURE);
        }

        if (vm.count("threads") && cfg_num_threads > 1) {
          myprintf("Nonsensical options: lczero loses deterministic property "
                   "of the random seed when using multiple threads.\n");
          exit(EXIT_FAILURE);
        }

        if (cfg_num_threads > 1) {
            cfg_num_threads = 1;
            myprintf("Using rng seed from cli, activating single thread mode!\n");
        }
        myprintf("RNG seed from cli: %llu\n", cfg_rng_seed);
    }

    if (vm.count("uci")) {
        cfg_noinitialize = true;
    }

    if (vm.count("noise")) {
        cfg_noise = true;
    }

    if (vm.count("randomize")) {
        cfg_randomize = true;
        // When cfg_randomize is on, we need an accurate estimate of
        // how good/bad all moves are, so turn cfg_timemanage off.
        cfg_timemanage = false;
    }

    if (vm.count("tempdecay")) {
        cfg_root_temp_decay = vm["tempdecay"].as<int>();
        if (cfg_root_temp_decay < 0) {
            myprintf("Nonsensical options: The temperature decay constant cannot be assigned a negative value, since that would turn the search useless in later game.\n");
            exit(EXIT_FAILURE);
        }
        cfg_randomize = true;
        // Setting a value for temperature decay constant also activates --randomize.
        // However, time management is not deactivated by --tempdecay
    }

    if (vm.count("playouts")) {
        cfg_max_playouts = vm["playouts"].as<int>();
        if (!vm.count("visits") && !vm.count("nodes")) {
            // If the user specifies playouts they probably
            // do not want the default 800 nodes.
            cfg_max_nodes = MAXINT_DIV2;
        }
    }

    if (vm.count("visits")) {
        cfg_max_nodes = vm["visits"].as<int>();
    }
    // let deprecated spelling be overwritten by new/correct spelling
    if (vm.count("nodes")) {
        cfg_max_nodes = vm["nodes"].as<int>();
    }

    if (vm.count("resignpct")) {
        cfg_resignpct = vm["resignpct"].as<int>();
    }

#ifdef USE_OPENCL
    if (vm.count("gpu")) {
        cfg_gpus = vm["gpu"].as<std::vector<int> >();
    }

    if (vm.count("full-tuner")) {
        cfg_sgemm_exhaustive = true;
    }

    if (vm.count("tune-only")) {
        cfg_tune_only = true;
    }
#endif

    std::string start = "";
    if (vm.count("start")) {
        start = vm["start"].as<std::string>();
    }

    return start;
}

void test_pgn_parse() {
  std::string raw = R"EOM([Event "?"]
[Site "?"]
[Date "2018.01.08"]
[Round "1"]
[White "Stockfish 080118 64 BMI2"]
[Black "Stockfish 080118 64 BMI2"]
[Result "0-1"]
[ECO "C88"]
[Opening "Ruy Lopez"]
[Variation "Closed"]
[TimeControl "10+0.1"]
[PlyCount "148"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8.
c3 O-O 9. h3 Bb7 10. d4 Re8 11. Nbd2 Bf8 12. d5 Nb8 13. Nf1 Nbd7 14. N3h2
c6 15. dxc6 Bxc6 16. Bg5 Qc7 17. Qf3 a5 18. Ng4 a4 19. Bc2 Nxg4 20. Qxg4
Re6 21. Qf3 Qb7 22. Ne3 Nb6 23. Rad1 Nc4 24. Bh4 Nxb2 25. Rb1 a3 26. Re2
Ree8 27. Ng4 Re6 28. Ne3 Rc8 29. Kh1 Na4 30. Rb3 g6 31. Rxa3 Nc5 32. Nd5
Bxd5 33. exd5 Ree8 34. Re1 Ra8 35. Rxa8 Rxa8 36. Bb3 f5 37. Rb1 Be7 38.
Bxe7 Qxe7 39. Qe3 Qh4 40. Kg1 f4 41. Qe2 e4 42. Qg4 Qf6 43. h4 Kg7 44. c4
bxc4 45. Bxc4 Ra4 46. Bb3 Rd4 47. Re1 Rd2 48. Qg5 h6 49. Qg4 Qf5 50. Qxf5
gxf5 51. Kf1 Kf6 52. Re2 Rxe2 53. Kxe2 Nxb3 54. axb3 Ke5 55. g3 f3+ 56. Ke3
h5 57. Kd2 Kxd5 58. g4 fxg4 59. Ke3 Kc5 60. b4+ Kxb4 61. Kxe4 d5+ 62. Ke5
g3 63. fxg3 f2 64. Kxd5 f1=Q 65. Ke5 Qf3 66. Ke6 Qxg3 67. Kf6 Kc5 68. Kf5
Qxh4 69. Kg6 Qg4+ 70. Kf7 Kd6 71. Kf6 h4 72. Kf7 Qg5 73. Kf8 Ke6 74. Ke8
Qe7# 0-1

[Event "?"]
)EOM";

  std::istringstream ss(raw);
  PGNParser parser(ss);
  auto game = parser.parse();

  if (game->bh.cur().fen() != "4K3/4q3/4k3/8/7p/8/8/8 w - - 6 75") {
    throw std::runtime_error("PGNParser fen broken");
  }
  if (game->result != -1) {
    throw std::runtime_error("PGNParser result broken: " + std::to_string(game->result));
  }
}

void generate_supervised_data(const std::string& filename) {
  namespace fs = boost::filesystem;
  fs::path fp(filename);
  fs::path dir("supervise-" + fp.stem().string());
  if (!fs::exists(dir)) {
    fs::create_directories(dir);
    myprintf_so("Created dirs %s\n", dir.string().c_str());
  }
  auto chunker = OutputChunker{dir.string() + "/training", true, 15000};

  std::ifstream f;
  f.open(filename);

  PGNParser parser(f);
  int games = 0;
  for (;;) {
    Training::clear_training();
    auto game = parser.parse();
    if (game == nullptr) {
      myprintf_so("Invalid game in %s\n", filename.c_str());
      break;
    }
    myprintf_so("\rProcessed %d games", ++games);
    BoardHistory bh;
    bh.set(Position::StartFEN);
    for (int i = 0; i < static_cast<int>(game->bh.positions.size()) - 1; ++i) {
      Move move = game->bh.positions[i + 1].get_move();
      Training::record(bh, move);
      bh.do_move(move);
    }
    Training::dump_training(game->result, chunker);
  }
}

int main(int argc, char* argv[]) {

  Bitboards::init();
  Position::init();

  Parameters::setup_default_parameters();
  std::string uci_start = parse_commandline(argc, argv);

  // test_pgn_parse();

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
  // Random::GetRng().seedrandom(cfg_rng_seed);
  if (!cfg_noinitialize) {
      Network::initialize();
  }

  if (!cfg_supervise.empty()) {
      generate_supervised_data(cfg_supervise);
      return 0;
  }

  Tablebases::init(cfg_syzygypath);
  UCI::init(Options);
  UCI::loop(uci_start);

  return 0;
}
