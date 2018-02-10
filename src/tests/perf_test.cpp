#include <gtest/gtest.h>

class PerfTest: public ::testing::Test {
public:
  PerfTest() {
  }
};

TEST_F(PerfTest, Init) {
  EXPECT_EQ(1, 1);
}
