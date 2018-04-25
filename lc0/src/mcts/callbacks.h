/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <functional>
#include "chess/bitboard.h"
#ifdef _MSC_VER
#include "utils/optional.h"
#else
#include <optional>
using std::optional;
#endif
#include <string>
#include <vector>

namespace lczero {

// Is sent when search decides on the best move.
struct BestMoveInfo {
  BestMoveInfo(Move bestmove, Move ponder = Move{})
      : bestmove(bestmove), ponder(ponder) {}
  Move bestmove;
  Move ponder;
  // Those are extensions and not really UCI protocol.
  // 1 if it's "player1", 2 if it's "player2"
  int player = -1;
  // Index of the game in the tournament (0-based).
  int game_id = -1;
  // The color of the player, if known.
  optional<bool> is_black;

  using Callback = std::function<void(const BestMoveInfo&)>;
};

// Is sent during the search.
struct ThinkingInfo {
  // Full depth.
  int depth = -1;
  // Maximum depth.
  int seldepth = -1;
  // Time since start of thinking.
  int64_t time = -1;
  // Nodes visited.
  int64_t nodes = -1;
  // Nodes per second.
  int nps = -1;
  // Hash fullness * 1000
  int hashfull = -1;
  // Win in centipawns.
  optional<int> score;
  // Best line found. Moves are from perspective of white player.
  std::vector<Move> pv;
  // Freeform comment.
  std::string comment;

  // Those are extensions and not really UCI protocol.
  // 1 if it's "player1", 2 if it's "player2"
  int player = -1;
  // Index of the game in the tournament (0-based).
  int game_id = -1;
  // The color of the player, if known.
  optional<bool> is_black;

  using Callback = std::function<void(const ThinkingInfo&)>;
};

// Is sent when a single game is finished.
struct GameInfo {
  enum GameResult { UNDECIDED, WHITE_WON, DRAW, BLACK_WON };
  // Game result.
  GameResult game_result = UNDECIDED;
  // Name of the file with training data.
  std::string training_filename;
  // Game moves.
  std::vector<Move> moves;
  // Index of the game in the tournament (0-based).
  int game_id = -1;
  // The color of the player1, if known.
  optional<bool> is_black;

  using Callback = std::function<void(const GameInfo&)>;
};

// Is sent in the end of tournament and also during the tournament.
struct TournamentInfo {
  // Did tournament finish, so those results are final.
  bool finished = false;

  // Player1's [win/draw/lose] as [white/black].
  // e.g. results[2][1] is how many times player 1 lost as black.
  int results[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  using Callback = std::function<void(const TournamentInfo&)>;
};

}  // namespace lczero