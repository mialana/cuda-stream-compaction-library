#include <gtest/gtest.h>

#include "test_utils.h"

#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>

using namespace stream_compaction;
using enum common::eTimerDevice;

class ScanTest : public testing::TestWithParam<int>
{
protected:
    void SetUp() override
    {
        int n = GetParam();
        _source.resize(n, -1);
        _expected.resize(n, -1);
        _actual.resize(n, -1);
        fill_container_random(_source, n);
    }

    std::vector<int> _source{};
    std::vector<int> _expected{};
    std::vector<int> _actual{};
};

constexpr int kNUM_POT = 24;
constexpr std::array<int, kNUM_POT> kPOT_VALUES = []
{
    std::array<int, kNUM_POT> arr{};
    for (int i = 0; i < kNUM_POT; ++i)
        arr[i] = 1 << i;
    return arr;
}();

constexpr int kNUM_NPOT = kNUM_POT - 2;  // '1' and '2' will not have an NPOT test case
constexpr std::array<int, kNUM_NPOT> kNPOT_VALUES = []
{
    std::array<int, kNUM_NPOT> arr{};
    for (int i = 0, pot = 2; i < kNUM_NPOT; ++i, ++pot)
        arr[i] = (1 << pot) - 3;
    return arr;
}();

INSTANTIATE_TEST_SUITE_P(PowersOfTwo, ScanTest, testing::ValuesIn(kPOT_VALUES),
                         testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(NonPowersOfTwo, ScanTest, testing::ValuesIn(kNPOT_VALUES),
                         testing::PrintToStringParamName());

TEST_P(ScanTest, naiveScan)
{
    cpu::scan(GetParam(), _source.data(), _expected.data());
    cpu::get_timer().flush<CPU>();
    naive::scan_wrapper(GetParam(), kBLOCK_SIZE, _source.data(), _actual.data());
    naive::get_timer().flush<GPU>();

    ASSERT_EQ(_expected, _actual);
}
