#include <gtest/gtest.h>
#include "../Network.h"

static bool is_initialized = false;

class NetworkTest: public ::testing::Test {
public:
  NetworkTest() {
    if (!is_initialized) {
      is_initialized = true;
      Bitboards::init();
      Position::init();
      Network::init_move_map();
    }
  }

  typedef decltype(Network::new_move_lookup) MoveLookup;

  static MoveLookup &get_move_lookup() {
    return Network::new_move_lookup;
  }
};

static NetworkTest::MoveLookup &move_lookup = NetworkTest::get_move_lookup();

TEST_F(NetworkTest, Init) {
  EXPECT_EQ(1, 1);
}

static Move get_move_for_id(int id) {
  for (auto &entry : move_lookup) {
    if (entry.second == id) return entry.first;
  }

  throw std::runtime_error("Move not found");
}

TEST_F(NetworkTest, MovesAreFlippedCorrectly) {
  for (auto &entry : move_lookup) {
    auto move = entry.first;
    auto flipped_move = Network::flip_move(move);

    ASSERT_EQ(rank_of(from_sq(move)), 7 - rank_of(from_sq(flipped_move)));
    ASSERT_EQ(file_of(from_sq(move)), file_of(from_sq(flipped_move)));

    ASSERT_EQ(rank_of(to_sq(move)), 7 - rank_of(to_sq(flipped_move)));
    ASSERT_EQ(file_of(to_sq(move)), file_of(to_sq(flipped_move)));

    ASSERT_EQ(type_of(move), type_of(flipped_move));
    ASSERT_EQ(promotion_type(move), promotion_type(flipped_move));
  }
}

TEST_F(NetworkTest, WhiteMovesAreLookedUpCorrectly) {
  for (auto &entry : move_lookup) {
    auto expected_move = entry.first;
    auto looked_up_move = get_move_for_id(Network::lookup(entry.first, Color::WHITE));
    ASSERT_EQ(expected_move, looked_up_move);
  }
}

TEST_F(NetworkTest, BlackMovesAreLookedUpCorrectly) {
  for (auto &entry : move_lookup) {
    auto flipped_move = Network::flip_move(entry.first);

    auto expected_move = entry.first;
    auto looked_up_move = get_move_for_id(Network::lookup(flipped_move, Color::BLACK));

    ASSERT_EQ(expected_move, looked_up_move);
  }
}
