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

#include <boost/filesystem.hpp>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "Movegen.h"
#include "Parameters.h"
#include "pgn.h"
#include "Position.h"
#include "Misc.h"
#include "Training.h"
#include "UCI.h"
#include "UCTSearch.h"
#include "Utils.h"

using namespace std;
using namespace Utils;

namespace {

  // position() is called when engine receives the "position" UCI command.
  // The function sets up the position described in the given FEN string ("fen")
  // or the starting position ("startpos") and then makes the moves given in the
  // following move list ("moves").

  void position(BoardHistory& bh, istringstream& is) {

    Move m;
    string token, fen;

    is >> token;

    if (token == "startpos")
    {
        fen = Position::StartFEN;
        is >> token; // Consume "moves" token if any
    }
    else if (token == "fen")
        while (is >> token && token != "moves")
            fen += token + " ";
    else
        return;

    bh.set(fen);

    // Parse move list (if any)
    while (is >> token && (m = UCI::to_move(bh.cur(), token)) != MOVE_NONE)
        bh.do_move(m);
  }


  // setoption() is called when engine receives the "setoption" UCI command. The
  // function updates the UCI option ("name") to the given value ("value").

  void setoption(istringstream& is) {

    string token, name, value;

    is >> token; // Consume "name" token

    // Read option name (can contain spaces)
    while (is >> token && token != "value")
        name += string(" ", name.empty() ? 0 : 1) + token;

    // Read option value (can contain spaces)
    while (is >> token)
        value += string(" ", value.empty() ? 0 : 1) + token;

    myprintf_so("No such option: %s\n", name.c_str());
  }

  // called when receiving the 'perft Depth' command
  void uci_perft(BoardHistory& bh, istringstream& is) {
       int d;
       is >> d;

       Depth depth = Depth(d);
       uint64_t total = UCI::perft<true>(bh, depth);
       myprintf_so("Total: %lld\n", total);
  }

void printVersion() {
  myprintf_so("id name lczero " PROGRAM_VERSION "\nid author The LCZero Authors\nuciok\n");
}

// Return the score from the self-play game
int play_one_game(BoardHistory& bh) {
  auto search = std::make_unique<UCTSearch>(bh.shallow_clone());
  for (int game_ply = 0; game_ply < 450; ++game_ply) {
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
    Limits.startTime = now();
    Move move = search->think(bh.shallow_clone());

    myprintf_so("move played %s\n", UCI::move(move).c_str());
    bh.do_move(move);
  }

  // Game termination as draw
  return 0;
}

int play_one_game() {
  BoardHistory bh;
  bh.set(Position::StartFEN);

  Training::clear_training();
  int game_score = play_one_game(bh);

  myprintf_so("PGN\n%s\nEND\n", bh.pgn().c_str());
  myprintf_so("Score: %d\n", game_score);

  return game_score;
}

void generate_training_games(istringstream& is) {
  printVersion();

  namespace fs = boost::filesystem;
  std::string suffix;
  if (!(is >> suffix)) {
    suffix = "0";
  }
  int64_t num_games = std::numeric_limits<int64_t>::max();
  is >> num_games;

  fs::path dir("data-" + suffix);
  if (!fs::exists(dir)) {
    fs::create_directories(dir);
    myprintf_so("Created dirs %s\n", dir.string().c_str());
  }
  auto chunker = OutputChunker{dir.string() + "/training", true};
  for (int64_t i = 0; i < num_games; i++) {
    Training::dump_training_v2(play_one_game(), chunker);
  }
}

void bench() {
  std::string raw = R"EOM([Event "?"]
[Site "?"]
[Date "2018.01.14"]
[Round "1"]
[White "lc_new"]
[Black "lc_base"]
[Result "1/2-1/2"]
[ECO "A40"]
[Opening "Queen's pawn"]
[PlyCount "90"]
[TimeControl "inf"]

1. d4 e6 {0.51s} 2. f3 Qg5 3. c4 Qxg2 {0.50s} 4. Bxg2 Nh6 {0.51s} 5. Nc3
Nc6 {0.52s} 6. e4 Bd6 {0.50s} 7. Nge2 g6 {0.50s} 8. O-O Bxh2+ {0.50s} 9. Kh1 Ke7
10. Be3 {0.50s} Kf6 11. Qd2 {0.50s} Nxd4 12. Nxd4 {0.50s} Rf8 13. e5+ Ke7 14. c5
Bg3 15. f4 d6 16. cxd6+ Ke8 17. Kg1 Bd7 18. a4 Rd8 {0.50s} 19. a5 Ra8 {0.54s}
20. a6 {0.51s} Rd8 21. Ndb5 Ra8 22. Nd4 Rd8 23. Ndb5 {0.51s} Ra8

)EOM";

  std::istringstream ss(raw);
  PGNParser parser(ss);
  auto game = parser.parse();

  myprintf_so("%s\n", game->bh.cur().fen().c_str());

  /*
  Network::DebugRawData debug_data;
  auto r = Network::get_scored_moves(game->bh, &debug_data);

  FILE* f = fopen("/tmp/output", "w");
  fputs(debug_data.getJson().c_str(), f);
  fclose(f);
  */

  auto search = std::make_unique<UCTSearch>(game->bh.shallow_clone());
  auto save_cfg_timemanage = cfg_timemanage;
  cfg_timemanage = false;
  search->set_quiet(false);
  search->think(game->bh.shallow_clone());
  cfg_timemanage = save_cfg_timemanage;
}

} // namespace

// count number of legal nodes up to a given depth from a given position.
template<bool Root>
uint64_t UCI::perft(BoardHistory& bh, Depth depth) {

  StateInfo st;
  uint64_t cnt, nodes = 0;
  const bool leaf = (depth == 2 * ONE_PLY);

  for (const auto& m : MoveList<LEGAL>(bh.cur()))
  {
      if (Root && depth <= ONE_PLY)
          cnt = 1, nodes++;
      else
      {
          bh.cur().do_move(m, st);
          cnt = leaf ? MoveList<LEGAL>(bh.cur()).size() : UCI::perft<false>(bh, depth - ONE_PLY);
          nodes += cnt;
          bh.cur().undo_move(m);
      }
      if (Root) {
          myprintf_so("%s: %lld\n", UCI::move(m).c_str(), cnt);
      }
  }
  return nodes;
}


// go() is called when engine receives the "go" UCI command. The function sets
// the thinking time and other parameters from the input string, then starts
// the search.

void gohelper(UCTSearch & search, BoardHistory &bh) {
    Move move = search.think(bh.shallow_clone());

    bh.do_move(move);
    myprintf_so("bestmove %s\n", UCI::move(move).c_str());
}

void go(UCTSearch& search, BoardHistory& bh, istringstream& is) {

    Limits = LimitsType();
    string token;

    // TODO: See issue #287.
    //if ((is >> token) && token == "infinite") Limits.infinite = 1;
    //else Limits.infinite = 0;
    Limits.infinite = 0;

    do {
        if (token == "wtime")          is >> Limits.time[WHITE];
        else if (token == "btime")     is >> Limits.time[BLACK];
        else if (token == "winc")      is >> Limits.inc[WHITE];
        else if (token == "binc")      is >> Limits.inc[BLACK];
        else if (token == "movestogo") is >> Limits.movestogo;
        else if (token == "depth")     is >> Limits.depth;
        else if (token == "nodes")     is >> Limits.nodes;
        else if (token == "movetime")  is >> Limits.movetime;
    } while (is >> token);

    // TODO: See issue #287.
    //std::thread lol(gohelper, std::ref(search), std::ref(bh));
    //lol.detach();
    gohelper(search, bh);
}


/// UCI::loop() waits for a command from stdin, parses it and calls the appropriate
/// function. Also intercepts EOF from stdin to ensure gracefully exiting if the
/// GUI dies unexpectedly. When called with some command line arguments, e.g. to
/// run 'bench', once the command is executed the function returns immediately.
/// In addition to the UCI ones, also some additional debug commands are supported.

void UCI::loop(const std::string& start) {
  string token, cmd = start;
  BoardHistory bh;
  bh.set(Position::StartFEN);
  UCTSearch search (bh.shallow_clone());//std::make_unique<UCTSearch>(bh.shallow_clone());

  do {
      if (start.empty() && !getline(cin, cmd)) // Block here waiting for input or EOF
          cmd = "quit";

      log_input(cmd);
      istringstream is(cmd);
      token.clear(); // Avoid a stale if getline() returns empty or blank line
      is >> skipws >> token;

      if (token == "quit" || token == "exit") break;

      /*
      // The GUI sends 'ponderhit' to tell us the user has played the expected move.
      // So 'ponderhit' will be sent if we were told to ponder on the same move the
      // user has played. We should continue searching but switch from pondering to
      // normal search. In case Threads.stopOnPonderhit is set we are waiting for
      // 'ponderhit' to stop the search, for instance if max search depth is reached.
      if (    token == "quit"
          ||  token == "stop"
          || (token == "ponderhit" && Threads.stopOnPonderhit))
          Threads.stop = true;
      else if (token == "ponderhit")
          Threads.ponder = false; // Switch to normal search
      */

      if (token == "uci")             printVersion();
      else if (token == "setoption")  setoption(is);
      else if (token == "go")         go(search,bh,is);
      else if (token == "stop")       search.please_stop();
      else if (token == "perft")      uci_perft(bh, is);
      else if (token == "position")   position(bh, is);
      else if (token == "ucinewgame") ;
      else if (token == "isready") {
          Network::initialize();
          myprintf_so("readyok\n");
      }
      // Additional custom non-UCI commands, mainly for debugging
      else if (token == "train")   generate_training_games(is);
      else if (token == "bench")   bench();
      else if (token == "d" || token == "showboard") {
		  std::stringstream ss;
		  ss << bh.cur();
		  myprintf_so("%s\n", ss.str().c_str());
	  }
	  else if (token == "showfen") {
	      std::stringstream ss;
		  ss << bh.cur().fen();
		  myprintf_so("%s\n", ss.str().c_str());
	  }
	  else if (token == "showgame") {
		  std::string result;
		  for (const auto &p : bh.positions) {
			  if (result == "") {
			      result = " "; // first position has no move
			  } else {
				  result += UCI::move(p.get_move()) + " ";
			  }
		  }
		  myprintf_so("position startpos%s\n", result.c_str());
	  }
	  else if (token == "showpgn") myprintf_so("%s\n", bh.pgn().c_str());
	  else if (token == "undo") myprintf_so(bh.undo_move() ? "Undone\n" : "At first move\n");
	  else if (token == "usermove" || token == "play") {
		  std::string ms; is >> ms;
		  Move m = UCI::to_move(bh.cur(), ms);
		  if (m == MOVE_NONE) m = bh.cur().san_to_move(ms);
		  if (m != MOVE_NONE) {
			  bh.do_move(m);
			  myprintf_so("usermove %s\n", UCI::move(m).c_str());
		  }
		  else {
			  myprintf_so("Illegal move: %s\n", ms.c_str());
		  }
	  }
	  else if (UCI::to_move(bh.cur(), token) != MOVE_NONE) {
		  Move m = UCI::to_move(bh.cur(), token);
		  bh.do_move(m);
		  myprintf_so("usermove %s\n", UCI::move(m).c_str());
	  }
		  //else if (token == "eval")  sync_cout << Eval::trace(pos) << sync_endl;
		  else if (token != "quit") {
		  myprintf_so("Unknown command: %s\n", cmd.c_str());
	  }

  } while (start.empty()); // Command line args are one-shot
}

/// UCI::square() converts a Square to a string in algebraic notation (g1, a7, etc.)

std::string UCI::square(Square s) {
    return std::string{ char('a' + file_of(s)), char('1' + rank_of(s)) };
}

/// UCI::move() converts a Move to a string in coordinate notation (g1f3, a7a8q).
/// The only special case is castling, where we print in the e1g1 notation in
/// normal chess mode, and in e1h1 notation in chess960 mode. Internally all
/// castling moves are always encoded as 'king captures rook'.

string UCI::move(Move m) {
    
    Square from = from_sq(m);
    Square to = to_sq(m);
    
    if (m == MOVE_NONE)
        return "(none)";
    
    if (m == MOVE_NULL)
        return "0000";
    
    if (type_of(m) == CASTLING)
        to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));
    
    string move = UCI::square(from) + UCI::square(to);
    
    if (type_of(m) == PROMOTION)
        move += " pnbrqk"[promotion_type(m)];
    
    return move;
}


/// UCI::to_move() converts a string representing a move in coordinate notation
/// (g1f3, a7a8q) to the corresponding legal Move, if any.

Move UCI::to_move(const Position& pos, string const& str) {
    for (const auto& m : MoveList<LEGAL>(pos))
        if (str == UCI::move(m))
            return m;
    
    return MOVE_NONE;
}

