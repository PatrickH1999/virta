#ifndef FIELD_H
#define FIELD_H

#include <cstddef>
#include <vector>

namespace virta {

template<typename Real, typename Derived>
class Field {

public:
    size_t n;
    std::vector<Real> data;

    explicit Field(size_t n_) noexcept : n(n_), data(n_, Real(0)) {}
    explicit Field(size_t n_, Real value_) noexcept : n(n_), data(n_, Real(value_)) {}

    ~Field() = default;
    Field(const Field&) = default;
    Field(Field&&) noexcept = default;
    Field& operator=(const Field&) = default;
    Field& operator=(Field&&) noexcept = default;
};

} // namespace virta

#endif
