#pragma once

#include "Position.h"

struct PGNGame {
  BoardHistory bh;
  int result;
};

class PGNParser {
 public:
  PGNParser(std::istream& is);

  std::unique_ptr<PGNGame> parse();

 private:
  std::istream& is_;
};
