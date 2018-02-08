#include <gtest/gtest.h>

class NetworkTest: public ::testing::Test {
public:
  NetworkTest() {
  }
};

TEST_F(NetworkTest, Init) {
  EXPECT_EQ(1, 1);
}
