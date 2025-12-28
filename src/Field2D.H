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
    int gcm;

    /// Staggering.
    int stag;

    static constexpr size_t compute_ni(size_t ni_, int gcm_, int stag_) {
        return ni_ + 2 * gcm_ + (stag_ == 0);
    }

    static constexpr size_t compute_nj(size_t nj_, int gcm_, int stag_) {
        return nj_ + 2 * gcm_ + (stag_ == 1);
    }

    /**
     * @brief Construct a 2D field of size ni_ × nj_ initialized to @p value_.
     */
    Field2D(size_t ni_, size_t nj_, Real value_, BCStruct<Real, 2> BC_, int gcm_, int stag_) noexcept
    : ni(compute_ni(ni_, gcm_, stag_))
    , nj(compute_nj(nj_, gcm_, stag_))
    , BC(BC_)
    , gcm(gcm_)
    , Field<Real, Field2D<Real>>(compute_ni(ni_, gcm_, stag_) * compute_nj(nj_, gcm_, stag_), value_)
    , stag(stag_) {}

    /**
     * @brief Construct a 2D field of size ni_ × nj_ initialized to @p value_, but with stag set to default value of -1 (no staggering).
     */
    Field2D(size_t ni_, size_t nj_, Real value_, BCStruct<Real, 2> BC_, int gcm_) noexcept : Field2D(ni_, nj_, value_, BC_, gcm_, -1) {}

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
            this->setILowNeumann();
        }
        // iHi:
        if (BC.iHi.type == BCTag::Dirichlet) {
            this->setIHighDirichlet(BC.iHi.value);
        } else if (BC.iHi.type == BCTag::Neumann) {
            this->setIHighNeumann();
        }
        // jLo:
        if (BC.jLo.type == BCTag::Dirichlet) {
            this->setJLowDirichlet(BC.jLo.value);
        } else if (BC.jLo.type == BCTag::Neumann) {
            this->setJLowNeumann();
        }
        // jHi:
        if (BC.jHi.type == BCTag::Dirichlet) {
            this->setJHighDirichlet(BC.jHi.value);
        } else if (BC.jHi.type == BCTag::Neumann) {
            this->setJHighNeumann();
        }
    }
     
    inline void setILowDirichlet(Real value) {
        int iOffset = (stag == 0) ? 0 : 1;
        int iMirr;
        parallel_for(Range<int>(0, gcm), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
            iMirr = 2 * gcm + iOffset - i;
            this->operator()(i, j) = 2 * value - this->operator()(iMirr, j);
        });
    }

    inline void setIHighDirichlet(Real value) {
        int iOffset = (stag == 0) ? 3 : 2;
        int iMirr;
        parallel_for(Range<int>(ni - gcm, ni), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
            iMirr = 2 * (ni - gcm) - iOffset - i;
            this->operator()(i, j) = 2 * value - this->operator()(iMirr, j);
        });
    }
    
    inline void setJLowDirichlet(Real value) {
        int jOffset = (stag == 1) ? 0 : 1;
        int jMirr;
        parallel_for(Range<int>(gcm, ni - gcm), Range<int>(0, gcm), [&] (int i, int j) {
            jMirr = 2 * gcm + jOffset - j;
            this->operator()(i, j) = 2 * value - this->operator()(i, jMirr);
        });
    }
    
    inline void setJHighDirichlet(Real value) {
        int jOffset = (stag == 1) ? 3 : 2;
        int jMirr;
        parallel_for(Range<int>(gcm, ni - gcm), Range<int>(nj - gcm, nj), [&] (int i, int j) {
            jMirr = 2 * (nj - gcm) - jOffset - j;
            this->operator()(i, j) = 2 * value - this->operator()(i, jMirr);
        });
    }
        
    inline void setILowNeumann() {
        int iOffset = (stag == 0) ? 1 : 0;
        int iMirr;
        parallel_for(Range<int>(0, gcm), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
            iMirr = 2 * gcm + iOffset - i;
            this->operator()(i, j) = this->operator()(iMirr, j);
        });
    }

    inline void setIHighNeumann() {
        int iOffset = (stag == 0) ? 3 : 2;
        int iMirr;
        parallel_for(Range<int>(ni - gcm, ni), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
            iMirr = 2 * (ni - gcm) - iOffset - i;
            this->operator()(i, j) = this->operator()(iMirr, j);
        });
    }
    
    inline void setJLowNeumann() {
        int jOffset = (stag == 1) ? 0 : 1;
        int jMirr;
        parallel_for(Range<int>(gcm, ni - gcm), Range<int>(0, gcm), [&] (int i, int j) {
            jMirr = 2 * gcm + jOffset - j;
            this->operator()(i, j) = this->operator()(i, jMirr);
        });
    }
    
    inline void setJHighNeumann() {
        int jOffset = (stag == 1) ? 3 : 2;
        int jMirr;
        parallel_for(Range<int>(gcm, ni - gcm), Range<int>(nj - gcm, nj), [&] (int i, int j) {
            jMirr = 2 * (nj - gcm) - jOffset - j;
            this->operator()(i, j) = this->operator()(i, jMirr);
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
