#include <gtest/gtest.h>

#include "Bitboard.h"
#include "Position.h"

class PositionTest: public ::testing::Test {
protected:
  static void SetUpTestCase() {
    Bitboards::init();
    Position::init();
  }
};

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
