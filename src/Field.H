#ifndef FIELD_H
#define FIELD_H

#include <cstddef>
#include <vector>

namespace virta {

/**
 * @brief Base class for field containers.
 *
 * @tparam Real   Floating-point type used for storage (float, double, ...).
 * @tparam Derived CRTP type of the derived field.
 */
template<typename Real, typename Derived>
class Field {

public:
    /**
     * @brief Total number of elements stored in the field.
     *
     * Always equal to the product of all grid dimensions in derived classes.
     */
    size_t n;

    /**
     * @brief Flat contiguous storage for the field data.
     */
    std::vector<Real> data;

    /**
     * @brief Construct a field with @p n elements initialized to zero.
     */
    explicit Field(size_t n_) noexcept : n(n_), data(n_, Real(0)) {}

    /**
     * @brief Construct a field with @p n elements initialized to @p value_.
     */
    explicit Field(size_t n_, Real value_) noexcept : n(n_), data(n_, Real(value_)) {}

    /// Default destructor.
    ~Field() = default;

    /// Copy constructor.
    Field(const Field&) = default;

    /// Move constructor.
    Field(Field&&) noexcept = default;

    /// Copy assignment.
    Field& operator=(const Field&) = default;

    /// Move assignment.
    Field& operator=(Field&&) noexcept = default;
};

} // namespace virta

#endif
