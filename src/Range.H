#ifndef RANGE_H
#define RANGE_H

#include <cstddef>

namespace virta {

/**
 * @brief Simple arithmetic iterator range.
 *
 * @tparam T Signed/unsigned integer type.
 */
template <typename T>
struct Range {

    /// Start index (inclusive).
    T begin;

    /// End index (exclusive).
    T end;

    /// Loop stride.
    T step;

    /**
     * @brief Construct a new Range.
     *
     * @param begin_ First index.
     * @param end_   One-past-the-end index.
     * @param step_  Loop stride (default = 1).
     */
    constexpr Range(T begin_, T end_, T step_ = 1) noexcept
        : begin(begin_), end(end_), step(step_) {}
};

} // namespace virta

#endif
