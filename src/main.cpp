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

extern const char* StartFEN;

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
int selfPlayResult(BoardHistory& bh) {
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

void selfPlayGame() {
  BoardHistory bh;
  bh.positions.emplace_back();
  bh.states.emplace_back(new StateInfo());
  bh.cur().set(StartFEN, bh.states.back().get());
  Training::clear_training();

  int game_score = selfPlayResult(bh);

  Training::dump_training(game_score, "training");
}

int main(int argc, char* argv[]) {

  Bitboards::init();
  Position::init();

  Parameters::setup_default_parameters();

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

  selfPlayGame();

  return 0;
}
