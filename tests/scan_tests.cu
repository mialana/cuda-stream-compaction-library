#include "gtest/gtest.h"
#include "test_utils.h"

#include <array>

class ScanTest : public testing::Test
{
protected:
    ScanTest()
    {
        fill_array_random(a, 64);
        fill_array_random(b, 64);
    }

    std::array<int, kSIZE> a{};
    std::array<int, kSIZE> b{};
};
