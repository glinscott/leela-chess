#include "pgn.h"

#include <istream>

PGNParser::PGNParser(std::istream& is)
  : is_(is) {
}

std::unique_ptr<BoardHistory> PGNParser::parse() {
  // Skip all the PGN headers
  std::string s;
  for (;;) {
    getline(is_, s);
    if (s.empty()) {
      break;
    }
  }

  std::unique_ptr<BoardHistory> bh(new BoardHistory);
  bh->set(Position::StartFEN);

  // Read in the moves
  for (int i = 0;; ++i) {
    is_ >> s;
    if (s.empty() || s == "[Event") {
      break;
    }

    // Skip the move numbers
    if ((i % 3) == 0) {
     continue;
    }

    Move m = bh->cur().san_to_move(s);
    if (m == MOVE_NONE) {
      throw std::runtime_error("Unable to parse pgn move " + s);
    }
    bh->do_move(m);
  }

  return bh;
}
