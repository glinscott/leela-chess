#include <gtest/gtest.h>

#include "Bitboard.h"
#include "Position.h"
#include "UCI.h"

class PositionTest: public ::testing::Test {
protected:
  static void SetUpTestCase() {
    Bitboards::init();
    Position::init();
  }
};

void mock_shallow_repetitions(BoardHistory&& bh, int rep_cnt) {
  EXPECT_EQ(bh.cur().repetitions_count(), rep_cnt);
}

TEST_F(PositionTest, IsDrawStartPosition) {
  Position pos;
  StateInfo si;
  pos.set(Position::StartFEN, &si);
  EXPECT_FALSE(pos.is_draw());
}

TEST_F(PositionTest, IsDrawBareKings) {
  Position pos;
  StateInfo si;
  pos.set("8/8/8/4k3/8/8/2K5/8 w - - 0 1", &si);
  EXPECT_TRUE(pos.is_draw());
}

TEST_F(PositionTest, IsDrawSingleMinorPiece) {
  Position pos;
  StateInfo si;
  pos.set("8/8/8/4k3/1N6/8/2K5/8 w - - 0 1", &si);
  EXPECT_TRUE(pos.is_draw());
  pos.set("8/8/8/4k3/7b/8/2K5/8 w - - 0 1", &si);
  EXPECT_TRUE(pos.is_draw());
}

TEST_F(PositionTest, IsDrawSingleMajorPieceOrPawn) {
  Position pos;
  StateInfo si;
  pos.set("8/8/8/4k3/8/5R2/2K5/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
  pos.set("8/8/8/4k3/8/5q2/2K5/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
  pos.set("8/8/8/4k3/8/5P2/2K5/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
}

TEST_F(PositionTest, IsDrawTwoKnights) {
  Position pos;
  StateInfo si;
  pos.set("8/8/8/3nk3/8/5N2/2K5/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
  pos.set("8/8/8/3nk3/8/5n2/2K5/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
}

TEST_F(PositionTest, IsDrawBishopAndKnight) {
  Position pos;
  StateInfo si;
  pos.set("8/8/8/3bk3/8/5N2/2K5/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
  pos.set("8/8/8/3Bk3/8/5N2/2K5/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
}

TEST_F(PositionTest, IsDrawMultipleBishopsSameColor) {
  Position pos;
  StateInfo si;
  pos.set("8/8/8/3Bk3/8/5B2/2K5/8 w - - 0 1", &si);
  EXPECT_TRUE(pos.is_draw());
  pos.set("8/8/8/4kb2/8/2K2B2/8/8 w - - 0 1", &si);
  EXPECT_TRUE(pos.is_draw());
  pos.set("B7/1B3b2/2B3b1/4k2b/8/8/2K5/8 w - - 0 1", &si);
  EXPECT_TRUE(pos.is_draw());
  pos.set("B7/1B6/2B5/4k3/8/8/2K5/8 w - - 0 1", &si);
  EXPECT_TRUE(pos.is_draw());
}

TEST_F(PositionTest, IsDrawMultipleBishopsNotSameColor) {
  Position pos;
  StateInfo si;
  pos.set("8/8/8/4k3/8/2K1bb2/8/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
  pos.set("8/8/8/4k3/8/2K1Bb2/8/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
  pos.set("B7/1B3b2/2B3b1/4k2b/7B/8/2K5/8 w - - 0 1", &si);
  EXPECT_FALSE(pos.is_draw());
}

TEST_F(PositionTest, KeyTest) {
  BoardHistory bh_;
  bh_.set(Position::StartFEN);
  bh_.do_move(UCI::to_move(bh_.cur(), "d2d4"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d7d5"));
  bh_.do_move(UCI::to_move(bh_.cur(), "e2e4"));
  auto key = bh_.cur().key();
  auto full_key = bh_.cur().full_key();

  // Normal transposition, both keys match
  bh_.set(Position::StartFEN);
  bh_.do_move(UCI::to_move(bh_.cur(), "e2e4"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d7d5"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d2d4"));
  EXPECT_TRUE(key == bh_.cur().key());
  EXPECT_TRUE(full_key == bh_.cur().full_key());

  // Shuffle knights, key matches but
  // full_key doesn't because of rule50 mismatch
  bh_.set(Position::StartFEN);
  bh_.do_move(UCI::to_move(bh_.cur(), "e2e4"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d7d5"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d2d4"));
  bh_.do_move(UCI::to_move(bh_.cur(), "g8f6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "g1f3"));
  bh_.do_move(UCI::to_move(bh_.cur(), "f6g8"));
  bh_.do_move(UCI::to_move(bh_.cur(), "f3g1"));
  EXPECT_TRUE(key == bh_.cur().key());
  EXPECT_FALSE(full_key == bh_.cur().full_key());

  // Different setup
  bh_.set(Position::StartFEN);
  bh_.do_move(UCI::to_move(bh_.cur(), "d2d4"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d7d5"));
  bh_.do_move(UCI::to_move(bh_.cur(), "c1d2"));
  bh_.do_move(UCI::to_move(bh_.cur(), "c8d7"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d2c1"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d7c8"));
  EXPECT_EQ(bh_.cur().repetitions_count(), 1);
  bh_.do_move(UCI::to_move(bh_.cur(), "c1d2"));
  bh_.do_move(UCI::to_move(bh_.cur(), "c8d7"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d2c1"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d7c8"));
  EXPECT_EQ(bh_.cur().repetitions_count(), 2);
  EXPECT_EQ(bh_.cur().rule50_count(), 8);
  key = bh_.cur().key();
  full_key = bh_.cur().full_key();

  // rule50 matches but repetitions_count doesn't
  bh_.set(Position::StartFEN);
  bh_.do_move(UCI::to_move(bh_.cur(), "d2d4"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d7d5"));
  bh_.do_move(UCI::to_move(bh_.cur(), "c1d2"));
  bh_.do_move(UCI::to_move(bh_.cur(), "c8d7"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d2c1"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d7e6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "c1d2"));
  bh_.do_move(UCI::to_move(bh_.cur(), "e6d7"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d2c1"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d7c8"));
  EXPECT_EQ(bh_.cur().repetitions_count(), 1);
  EXPECT_EQ(bh_.cur().rule50_count(), 8);
  EXPECT_TRUE(key == bh_.cur().key());
  EXPECT_FALSE(full_key == bh_.cur().full_key());

  // Longer repetition test with shallow_clone
  bh_.set(Position::StartFEN);
  bh_.do_move(UCI::to_move(bh_.cur(), "a2a4")); bh_.do_move(UCI::to_move(bh_.cur(), "h7h5"));
  bh_.do_move(UCI::to_move(bh_.cur(), "a1a2")); bh_.do_move(UCI::to_move(bh_.cur(), "h8h6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "a2a3")); bh_.do_move(UCI::to_move(bh_.cur(), "h6a6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "a3b3")); bh_.do_move(UCI::to_move(bh_.cur(), "a6b6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "b3c3")); bh_.do_move(UCI::to_move(bh_.cur(), "b6c6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "c3d3")); bh_.do_move(UCI::to_move(bh_.cur(), "c6d6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "d3e3")); bh_.do_move(UCI::to_move(bh_.cur(), "d6e6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "e3f3")); bh_.do_move(UCI::to_move(bh_.cur(), "e6f6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "f3g3")); bh_.do_move(UCI::to_move(bh_.cur(), "f6g6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "g3h3"));
  EXPECT_EQ(bh_.cur().repetitions_count(), 0);
  mock_shallow_repetitions(bh_.shallow_clone(), 0);
  bh_.do_move(UCI::to_move(bh_.cur(), "g6h6"));
  EXPECT_EQ(bh_.cur().repetitions_count(), 0);
  mock_shallow_repetitions(bh_.shallow_clone(), 0);
  bh_.do_move(UCI::to_move(bh_.cur(), "h3a3"));
  EXPECT_EQ(bh_.cur().repetitions_count(), 1);
  mock_shallow_repetitions(bh_.shallow_clone(), 1);
  bh_.do_move(UCI::to_move(bh_.cur(), "h6a6"));
  EXPECT_EQ(bh_.cur().repetitions_count(), 1);
  mock_shallow_repetitions(bh_.shallow_clone(), 1);
}

TEST_F(PositionTest, PGNTest) {
  BoardHistory bh_;
  bh_.set(Position::StartFEN);
  bh_.do_move(UCI::to_move(bh_.cur(), "f2f4"));
  bh_.do_move(UCI::to_move(bh_.cur(), "a7a6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "f4f5"));
  bh_.do_move(UCI::to_move(bh_.cur(), "e7e6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "f5e6"));
  EXPECT_EQ(bh_.pgn(), "1. f4 a6 2. f5 e6 3. fxe6 ");

  bh_.set(Position::StartFEN);
  bh_.do_move(UCI::to_move(bh_.cur(), "f2f4"));
  bh_.do_move(UCI::to_move(bh_.cur(), "a7a6"));
  bh_.do_move(UCI::to_move(bh_.cur(), "f4f5"));
  bh_.do_move(UCI::to_move(bh_.cur(), "e7e5"));
  bh_.do_move(UCI::to_move(bh_.cur(), "f5e6"));
  EXPECT_EQ(bh_.pgn(), "1. f4 a6 2. f5 e5 3. fxe6 ");
}
