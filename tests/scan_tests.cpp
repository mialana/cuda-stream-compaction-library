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

constexpr int kMAX_POT = 28;
constexpr std::array<int, kMAX_POT> kPOT_VALUES = []
{
    std::array<int, kMAX_POT> arr{};
    for (int i = 0; i < kMAX_POT; ++i)
        arr[i] = 1 << i;
    return arr;
}();

INSTANTIATE_TEST_SUITE_P(PowersOfTwo, ScanTest, testing::ValuesIn(kPOT_VALUES),
                         testing::PrintToStringParamName());

TEST_P(ScanTest, naiveScanPowerOfTwo)
{
    cpu::scan(GetParam(), _source.data(), _expected.data());
    cpu::get_timer().flush<CPU>();
    naive::scan(GetParam(), _source.data(), _actual.data());
    naive::get_timer().flush<GPU>();

    for (int i = 0; i < GetParam(); ++i)
    {
        ASSERT_EQ(_expected[i], _actual[i]) << "(Index " << i << ")";
    }
}
