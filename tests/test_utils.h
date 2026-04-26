#pragma once

#include <random>
#include <array>

constexpr int kSIZE = 1 << 24;
constexpr int kNPOT = kSIZE - 3;  // non-power-of-two

#define POW2(x) (1 << (x))

template<typename T>
concept ContainerIntegral
    = std::ranges::contiguous_range<T>  // is contiguous
      && std::integral<std::ranges::range_value_t<T>>  // has element of integral type
      && std::is_standard_layout_v<std::ranges::range_value_t<T>>  // has C-compatible memory layout
      && requires(T t) {
             { t.data() } -> std::same_as<std::ranges::range_value_t<T>*>;  // has `data()` method
             { t.size() } -> std::convertible_to<std::size_t>;  // has `size()` method
         };

template<ContainerIntegral T>
void fill_container_random(T& ctn, int max_val)
{
    using Element = std::ranges::range_value_t<T>;

    std::random_device rd;
    std::mt19937 gen(rd());  // initialize Mersenne Twister engine
    std::uniform_int_distribution<Element> distrib(0, max_val);

    for (int i = 0; i < ctn.size(); ++i)
        ctn[i] = distrib(gen);
}

template<ContainerIntegral T>
void print_container(const T& ctn, bool abridged = true)
{
    int n = static_cast<int>(ctn.size());
    int max_size = abridged ? std::min(n, 16) : n;

    std::cout << '\t' << "[ ";
    for (int i = 0; i < max_size; ++i)
        printf("%i ", static_cast<int>(ctn[i]));
    if (abridged && max_size < n) std::cout << "...";
    printf(" ] - count: %i\n", n);
}
