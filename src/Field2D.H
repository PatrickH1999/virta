#ifndef FIELD2D_H
#define FIELD2D_H

#include <cstddef>
#include <fstream>
#include <string>

#include "Field.H"

namespace virta {

/**
 * @brief A 2D scalar field stored in row-major format.
 *
 * @tparam Real The floating-point type of the stored data (e.g. float, double).
 *
 * Field2D represents a two-dimensional grid with ni × nj elements.
 * The underlying storage is a flat std::vector<Real> inherited from Field,
 * with row-major indexing: index = j * ni + i.
 *
 * This class provides element access through operator() for both 1D (raw index)
 * and 2D (i,j) addressing.
 */
template<typename Real>
class Field2D : public Field<Real, Field2D<Real>> {

public:
    /**
     * @brief Number of grid points in x-direction.
     */
    std::size_t ni;

    /**
     * @brief Number of grid points in y-direction.
     */
    std::size_t nj;

    /**
     * @brief Dimensionality of the field (always 2).
     */
    static constexpr int ndim = 2;

    /**
     * @brief Construct a 2D field with dimensions (ni_, nj_).
     *
     * All values are initialized to zero.
     *
     * @param ni_ Number of points in x-direction.
     * @param nj_ Number of points in y-direction.
     */
    Field2D(std::size_t ni_, std::size_t nj_) noexcept
        : Field<Real, Field2D<Real>>(ni_ * nj_), ni(ni_), nj(nj_) {}

    /**
     * @brief Construct a 2D field with dimensions (ni_, nj_) and fill with value_.
     *
     * @param ni_ Number of points in x-direction.
     * @param nj_ Number of points in y-direction.
     * @param value_ Initial fill value for all elements.
     */
    Field2D(std::size_t ni_, std::size_t nj_, Real value_) noexcept
        : Field<Real, Field2D<Real>>(ni_ * nj_, value_), ni(ni_), nj(nj_) {}

    /**
     * @brief Access a field element using a flat index.
     *
     * @param i Flat index in range [0, n).
     * @return Reference to the data element.
     */
    inline Real& operator()(std::size_t i) noexcept {
        return this->data[i];
    }

    /**
     * @brief Const-access a field element using a flat index.
     *
     * @param i Flat index in range [0, n).
     * @return Const reference to the data element.
     */
    inline const Real& operator()(std::size_t i) const noexcept {
        return this->data[i];
    }

    /**
     * @brief Access a field element using 2D indices (i, j).
     *
     * @param i Index in x-direction, 0 ≤ i < ni.
     * @param j Index in y-direction, 0 ≤ j < nj.
     * @return Reference to the data element.
     */
    inline Real& operator()(std::size_t i, std::size_t j) noexcept {
        return this->data[j * ni + i];
    }

    /**
     * @brief Const-access a field element using 2D indices (i, j).
     *
     * @param i Index in x-direction.
     * @param j Index in y-direction.
     * @return Const reference to the data element.
     */
    inline const Real& operator()(std::size_t i, std::size_t j) const noexcept {
        return this->data[j * ni + i];
    }

    /**
     * @brief Write the 2D field to a binary file.
     *
     * The written file format is:
     *   - uint64: ni
     *   - uint64: nj
     *   - Real[ni*nj]: raw field data in row-major order
     *
     * @param filename Output filename.
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
