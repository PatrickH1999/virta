#ifndef FIELD1D_H
#define FIELD1D_H

#include "Field.H"

namespace virta {

/**
 * @brief One-dimensional field container.
 *
 * @tparam Real Floating-point data type.
 */
template<typename Real>
class Field1D : public Field<Real, Field1D<Real>> {

public:
    /// Number of grid points in x-direction.
    size_t ni;

    /// Dimensionality of the field (1D).
    static constexpr int ndim = 1;

    /**
     * @brief Construct a 1D field of size @p ni_ initialized to zero.
     */
    Field1D(size_t ni_) noexcept
        : Field<Real, Field1D<Real>>(ni_), ni(ni_) {}

    /**
     * @brief Construct a 1D field of size @p ni_ initialized to @p value_.
     */
    Field1D(size_t ni_, Real value_) noexcept
        : Field<Real, Field1D<Real>>(ni_, value_), ni(ni_) {}

    /**
     * @brief Access element @p i (mutable).
     */
    inline Real& operator()(size_t i) noexcept {
        return this->data[i];
    }

    /**
     * @brief Access element @p i (const).
     */
    inline const Real& operator()(size_t i) const noexcept {
        return this->data[i];
    }
};

} // namespace virta

#endif
