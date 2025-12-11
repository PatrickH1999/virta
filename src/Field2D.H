#ifndef FIELD2D_H
#define FIELD2D_H

#include <cstddef>
#include <fstream>
#include <string>

#include "Field.H"

namespace virta {

/**
 * @brief Two-dimensional field container.
 *
 * Storage is contiguous in row-major order: data[j * ni + i].
 *
 * @tparam Real Floating-point data type.
 */
template<typename Real>
class Field2D : public Field<Real, Field2D<Real>> {

public:
    /// Number of points in x-direction.
    size_t ni;

    /// Number of points in y-direction.
    size_t nj;

    /// Dimensionality of the field (2D).
    static constexpr int ndim = 2;

    /**
     * @brief Construct a 2D field of size ni_ × nj_ initialized to zero.
     */
    Field2D(size_t ni_, size_t nj_) noexcept
        : Field<Real, Field2D<Real>>(ni_ * nj_), ni(ni_), nj(nj_) {}

    /**
     * @brief Construct a 2D field of size ni_ × nj_ initialized to @p value_.
     */
    Field2D(size_t ni_, size_t nj_, Real value_) noexcept
        : Field<Real, Field2D<Real>>(ni_ * nj_, value_), ni(ni_), nj(nj_) {}

    /**
     * @brief Access flat index @p i (mutable).
     */
    inline Real& operator()(size_t i) noexcept {
        return this->data[i];
    }

    /**
     * @brief Access flat index @p i (const).
     */
    inline const Real& operator()(size_t i) const noexcept {
        return this->data[i];
    }

    /**
     * @brief Access element at (i, j) using row-major storage.
     */
    inline Real& operator()(size_t i, size_t j) noexcept {
        return this->data[j * ni + i];
    }

    /**
     * @brief Access element at (i, j) (const).
     */
    inline const Real& operator()(size_t i, size_t j) const noexcept {
        return this->data[j * ni + i];
    }

    /**
     * @brief Write the field to a binary file with header: ni, nj, raw data.
     */
    void write_binary(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&this->ni), sizeof(this->ni));
        out.write(reinterpret_cast<const char*>(&this->nj), sizeof(this->nj));
        out.write(reinterpret_cast<const char*>(this->data.data()),
                  this->data.size() * sizeof(Real));
    }
};

} // namespace virta

#endif
