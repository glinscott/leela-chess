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

#include "selfplay/loop.h"
#include "selfplay/tournament.h"

namespace lczero {

namespace {
const char* kInteractive = "Run in interactive mode with uci-like interface";
}  // namespace

SelfPlayLoop::SelfPlayLoop() {}

void SelfPlayLoop::RunLoop() {
  options_.Add<CheckOption>(kInteractive, "interactive") = false;
  SelfPlayTournament::PopulateOptions(&options_);

  if (!options_.ProcessAllFlags()) return;
  if (options_.GetOptionsDict().Get<bool>(kInteractive)) {
    UciLoop::RunLoop();
  } else {
    SelfPlayTournament tournament(
        options_.GetOptionsDict(),
        std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
        std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
    tournament.RunBlocking();
  }
}

void SelfPlayLoop::SendGameInfo(const GameInfo& info) {
  std::string res = "gameready";
  if (!info.training_filename.empty())
    res += " trainingfile " + info.training_filename;
  if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
  if (info.is_black)
    res += " player1 " + std::string(*info.is_black ? "black" : "white");
  if (info.game_result != GameInfo::UNDECIDED) {
    res += std::string(" result ") +
           ((info.game_result == GameInfo::DRAW)
                ? "draw"
                : (info.game_result == GameInfo::WHITE_WON) ? "whitewon"
                                                            : "blackwon");
  }
  if (!info.moves.empty()) {
    res += " moves";
    for (const auto& move : info.moves) res += " " + move.as_string();
  }
  SendResponse(res);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  std::string res = "tournamentstatus";
  if (info.finished) res += " final";
  res += " win " + std::to_string(info.results[0][0]) + " " +
         std::to_string(info.results[0][1]);
  res += " lose " + std::to_string(info.results[2][0]) + " " +
         std::to_string(info.results[2][1]);
  res += " draw " + std::to_string(info.results[1][0]) + " " +
         std::to_string(info.results[1][1]);
  SendResponse(res);
}

}  // namespace lczero