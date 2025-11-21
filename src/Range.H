#ifndef RANGE_H
#define RANGE_H

#include <cstddef>

namespace virta {

template <typename T>
struct Range {
    T begin;
    T end;
    T step;

    constexpr Range(T begin_, T end_, T step_ = 1) noexcept : begin(begin_), end(end_), step(step_) {}
};

} // namespace virta

#endif
