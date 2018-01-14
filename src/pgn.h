#pragma once

#include "Position.h"

class PGNParser {
 public:
  PGNParser(std::istream& is);

  std::unique_ptr<BoardHistory> parse(); 

 private:
  std::istream& is_;
};
