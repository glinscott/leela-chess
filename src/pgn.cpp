#include "pgn.h"

#include <boost/optional.hpp>
#include <istream>

PGNParser::PGNParser(std::istream& is)
  : is_(is) {
}

boost::optional<int> parse_result(const std::string& result) {
  if (result == "1-0") {
    return 1;
  } else if (result == "0-1") {
    return -1;
  } else if (result == "1/2-1/2") {
    return 0;
  }
  return boost::none;
}

std::unique_ptr<PGNGame> PGNParser::parse() {
  // Skip all the PGN headers
  std::string s;
  std::string result;
  const std::string kResultToken = "[Result \"";
  for (;;) {
    getline(is_, s);
    if (s.substr(0, kResultToken.size()) == kResultToken) {
      result = s.substr(kResultToken.size(), s.size() - kResultToken.size() - 2);
    }
    if (s.empty()) {
      break;
    }
  }

  std::unique_ptr<PGNGame> game(new PGNGame);
  game->bh.set(Position::StartFEN);

  auto game_result = parse_result(result);
  if (game_result) {
    game->result = game_result.get();
  } else {
    throw std::runtime_error("Unknown result: " + result);
  }

  // Read in the moves
  for (int i = 0;; ++i) {
    is_ >> s;
    if (s.empty() || s == "[Event" || parse_result(s)) {
      break;
    }

    // Skip the move numbers
    if ((i % 3) == 0) {
     continue;
    }

    Move m = game->bh.cur().san_to_move(s);
    if (m == MOVE_NONE) {
      throw std::runtime_error("Unable to parse pgn move " + s);
    }
    game->bh.do_move(m);
  }

  // Read the empty line after the game
  getline(is_, s);
  getline(is_, s);

  return game;
}
