#include <gtest/gtest.h>

#include "Types.h"
#include "Bitboard.h"
#include "Position.h"
#include "UCI.h"

class PerfTest: public ::testing::Test {
public:
  PerfTest() {
    Bitboards::init();
    Position::init();
  }

  BoardHistory bh_;
};

// FEN positions from https://chessprogramming.wikispaces.com/Perft+Results

TEST_F(PerfTest, InitPos) {
  bh_.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  auto n = UCI::perft<true>(bh_, Depth(4));
  EXPECT_EQ(n, 197'281);
}

TEST_F(PerfTest, Pos2) {
  bh_.set("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
  auto n = UCI::perft<true>(bh_, Depth(4));
  EXPECT_EQ(n, 4'085'603);
}

TEST_F(PerfTest, Pos3) {
  bh_.set("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -");
  auto n = UCI::perft<true>(bh_, Depth(6));
  EXPECT_EQ(n, 11'030'083);
}

TEST_F(PerfTest, Pos4) {
  bh_.set("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
  auto n = UCI::perft<true>(bh_, Depth(5));
  EXPECT_EQ(n, 15'833'292);
}

TEST_F(PerfTest, Pos5) {
  bh_.set("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
  auto n = UCI::perft<true>(bh_, Depth(4));
  EXPECT_EQ(n, 2'103'487);
}

TEST_F(PerfTest, Pos6) {
  bh_.set("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
  auto n = UCI::perft<true>(bh_, Depth(4));
  EXPECT_EQ(n, 3'894'594);
}

