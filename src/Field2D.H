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

    /**
     * @brief Construct a 2D field of size ni_ × nj_ initialized to @p value_.
     */
    Field2D(size_t ni_, size_t nj_, Real value_, BCStruct<Real, 2> BC_, int gcm_, int stag_) noexcept : ni(ni_), nj(nj_), BC(BC_), gcm(gcm_), Field<Real, Field2D<Real>>(ni_ * nj_, value_), stag(stag_) {}

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
        parallel_for(Range<int>(0, gcm), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
            this->operator()(i, j) = value;
        });
    }
    
    inline void setIHighDirichlet(Real value) {
        parallel_for(Range<int>(ni - gcm, ni), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
            this->operator()(i, j) = value;
        });
    }
    
    inline void setJLowDirichlet(Real value) {
        parallel_for(Range<int>(gcm, ni - gcm), Range<int>(0, gcm), [&] (int i, int j) {
            this->operator()(i, j) = value;
        });
    }
    
    inline void setJHighDirichlet(Real value) {
        parallel_for(Range<int>(gcm, ni - gcm), Range<int>(nj - gcm, nj), [&] (int i, int j) {
            this->operator()(i, j) = value;
        });
    }
   
    /*
    gcm = 2
    i = 0 <=> i = 3
    i = 1 <=> i = 2

    gcm = 1
    i = 0 <=> i = 1

    gcm = 3
    i = 0 <=> i = 5
    i = 1 <=> i = 4
    i = 2 <=> i = 3

    gcm = 4
    i = 0 <=> j = 7
    i = 1 <=> j = 6
    i = 2 <=> j = 5
    i = 3 <=> j = 4

    j = 2 * gcm - 1 - i
    */
        
    inline void setILowNeumann() {
        if (stag == -1) {
            int i_mirr;
            parallel_for(Range<int>(0, gcm), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
                i_mirr = 2 * gcm - 1 - i;
                this->operator()(i, j) = this->operator()(i_mirr, j);
            });
        }
    }
    
    /*
    gcm = 4
    ni = 100
    i = ni - 4 <=> j = ni - 5 = i - 1
    i = ni - 3 <=> j = ni - 6 = i - 3
    i = ni - 2 <=> j = ni - 7 = i - 5
    i = ni - 1 <=> j = ni - 8 = i - 7
    j = i - 2 * (i - ni + gcm) - 1 = i - 2*i + 2*ni - 2*gcm - 1 = -i + 2 * (ni - gcm) - 1
    */

    inline void setIHighNeumann() {
        if (stag == -1) {
            int i_mirr;
            parallel_for(Range<int>(ni - gcm, ni), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
                i_mirr = 2 * (ni - gcm) - 1 - i;
                this->operator()(i, j) = this->operator()(i_mirr, j);
            });
        }
    }
    
    inline void setJLowNeumann() {
        if (stag == -1) {
            int j_mirr;
            parallel_for(Range<int>(gcm, ni - gcm), Range<int>(0, gcm), [&] (int i, int j) {
                j_mirr = 2 * gcm - 1 - j;
                this->operator()(i, j) = this->operator()(i, j_mirr);
            });
        }
    }
    
    inline void setJHighNeumann() {
        if (stag == -1) {
            int j_mirr;
            parallel_for(Range<int>(gcm, ni - gcm), Range<int>(nj - gcm, nj), [&] (int i, int j) {
                j_mirr = 2 * (nj - gcm) - 1 - j;
                this->operator()(i, j) = this->operator()(i, j_mirr);
            });
        }
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
