#ifndef FIELD2D_H
#define FIELD2D_H

#include <cstddef>
#include <fstream>
#include <string>

#include "BCStruct.H"
#include "Field.H"
#include "Range.H"

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
    /// Dimensionality of the field (2D).
    static constexpr int ndim = 2;
    
    /// Number of points in i-direction.
    size_t ni;

    /// Number of points in y-direction.
    size_t nj;

    /// Boundary conditions.
    BCStruct<Real, 2> BC;

    /// Ghost cell margin.
    int bcm;

    /// Staggering.
    int stag;

    /**
     * @brief Construct a 2D field of size ni_ × nj_ initialized to @p value_.
     */
    Field2D(size_t ni_, size_t nj_, Real value_, BCStruct<Real, 2> BC_, int bcm_, int stag_) noexcept : ni(ni_), nj(nj_), BC(BC_), bcm(bcm_), Field<Real, Field2D<Real>>(ni_ * nj_, value_), stag(stag_) {}

    /**
     * @brief Construct a 2D field of size ni_ × nj_ initialized to @p value_, but with stag set to default value of -1 (no staggering).
     */
    Field2D(size_t ni_, size_t nj_, Real value_, BCStruct<Real, 2> BC_, int bcm_) noexcept : Field2D(ni_, nj_, value_, BC_, bcm_, -1) {}

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

    inline void setBC() {
        // iLo:
        if (BC.iLo.type == BCTag::Dirichlet) {
            this->setILowDirichlet(BC.iLo.value);
        } else if (BC.iLo.type == BCTag::Neumann) {
            this->setILowNeumann(bcm);
        }
        // iHi:
        if (BC.iHi.type == BCTag::Dirichlet) {
            this->setIHighDirichlet(BC.iHi.value);
        } else if (BC.iHi.type == BCTag::Neumann) {
            this->setIHighNeumann(bcm);
        }
        // jLo:
        if (BC.jLo.type == BCTag::Dirichlet) {
            this->setJLowDirichlet(BC.jLo.value);
        } else if (BC.jLo.type == BCTag::Neumann) {
            this->setJLowNeumann(bcm);
        }
        // jHi:
        if (BC.jHi.type == BCTag::Dirichlet) {
            this->setJHighDirichlet(BC.jHi.value);
        } else if (BC.jHi.type == BCTag::Neumann) {
            this->setJHighNeumann(bcm);
        }
    }

    inline void setILowDirichlet(Real value) {
        parallel_for(Range<int>(0, bcm), Range<int>(bcm, nj - bcm), [&] (int i, int j) {
            this->operator()(i, j) = value;
        });
    }
    
    inline void setIHighDirichlet(Real value) {
        parallel_for(Range<int>(ni - bcm, ni), Range<int>(bcm, nj - bcm), [&] (int i, int j) {
            this->operator()(i, j) = value;
        });
    }
    
    inline void setJLowDirichlet(Real value) {
        parallel_for(Range<int>(bcm, ni - bcm), Range<int>(0, bcm), [&] (int i, int j) {
            this->operator()(i, j) = value;
        });
    }
    
    inline void setJHighDirichlet(Real value) {
        parallel_for(Range<int>(bcm, ni - bcm), Range<int>(nj - bcm, nj), [&] (int i, int j) {
            this->operator()(i, j) = value;
        });
    }
    
    inline void setILowNeumann(int bcm) {
        parallel_for(Range<int>(0, bcm), Range<int>(bcm, nj - bcm), [&] (int i, int j) {
            this->operator()(i, j) = this->operator()(bcm, j);
        });
    }
    
    inline void setIHighNeumann(int bcm) {
        parallel_for(Range<int>(ni - bcm, ni), Range<int>(bcm, nj - bcm), [&] (int i, int j) {
            this->operator()(i, j) = this->operator()(ni - bcm - 1, j);
        });
    }
    
    inline void setJLowNeumann(int bcm) {
        parallel_for(Range<int>(bcm, ni - bcm), Range<int>(0, bcm), [&] (int i, int j) {
            this->operator()(i, j) = this->operator()(i, bcm);
        });
    }
    
    inline void setJHighNeumann(int bcm) {
        parallel_for(Range<int>(bcm, ni - bcm), Range<int>(nj - bcm, nj), [&] (int i, int j) {
            this->operator()(i, j) = this->operator()(i, nj - bcm - 1);
        });
    }

    /**
     * @brief Write the field to a binary file with header: ni, nj, raw data.
     */
    void write_binary(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&ni), sizeof(ni));
        out.write(reinterpret_cast<const char*>(&nj), sizeof(nj));
        out.write(reinterpret_cast<const char*>(this->data.data()),
                  this->data.size() * sizeof(Real));
    }
};

} // namespace virta

#endif
