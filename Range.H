#ifndef RANGE_H
#define RANGE_H

namespace virta {

template <typename T>
struct Range {
    T begin;
    T end;
    T step;

    Range(T begin_, T end_, T step_ = 1) : begin(begin_), end(end_), step(step_) {}
};

} // namespace virta

#endif
