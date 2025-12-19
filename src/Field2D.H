#ifndef FIELD2D_H
#define FIELD2D_H

#include <cassert>
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

    BCStruct<Real, 2> BC;

    /// Ghost cell margin.
    int gcm;

    /// Staggering in i, j direction.
    bool stag_i;
    bool stag_j;

    /**
     * @brief Construct a 2D field of size ni_ × nj_ initialized to zero.
     */
    Field2D(size_t ni_, size_t nj_, BCStruct<Real, 2> BC_, int gcm_, bool stag_i_, bool stag_j_) noexcept : Field<Real, Field2D<Real>>(ni_times_nj(ni_, nj_, stag_i_, stag_j_)) {

        BC = BC_;
        stag_i = stag_i_;
        stag_j = stag_j_;
        assert(!(stag_i && stag_j)); // make sure that stag_i, stag_j are not both true
        if (stag_i) {
            ni = ni_ + 1;
            nj = nj_;
        } else if (stag_j) {
            ni = ni_;
            nj = nj_ + 1;
        } else {
            ni = ni_;
            nj = nj_;
        }
    }

    /**
     * @brief Construct a 2D field of size ni_ × nj_ initialized to @p value_.
     */
    Field2D(size_t ni_, size_t nj_, Real value_, BCStruct<Real, 2> BC_, int gcm_, bool stag_i_, bool stag_j_) noexcept : Field<Real, Field2D<Real>>(ni_times_nj(ni_, nj_, stag_i_, stag_j_), value_) {
        BC = BC_;
        stag_i = stag_i_;
        stag_j = stag_j_;
        assert(!(stag_i && stag_j)); // make sure that stag_i, stag_j are not both true
        if (stag_i) {
            ni = ni_ + 1;
            nj = nj_;
        } else if (stag_j) {
            ni = ni_;
            nj = nj_ + 1;
        } else {
            ni = ni_;
            nj = nj_;
        } 
    }

    size_t ni_times_nj(size_t ni_, size_t nj_, bool stag_i_, bool stag_j_) {
        assert(!(stag_i_ && stag_j_));
        if (stag_i_) {
            return ((ni_ + 1) * nj_);
        } else if (stag_j_) {
            return (ni_ * (nj_ + 1));
        } else {
            return (ni_ * nj_);
        }
    }

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
            this->setILowNeumann(gcm);
        }
        // iHi:
        if (BC.iHi.type == BCTag::Dirichlet) {
            this->setIHighDirichlet(BC.iHi.value);
        } else if (BC.iHi.type == BCTag::Neumann) {
            this->setIHighNeumann(gcm);
        }
        // jLo:
        if (BC.jLo.type == BCTag::Dirichlet) {
            this->setJLowDirichlet(BC.jLo.value);
        } else if (BC.jLo.type == BCTag::Neumann) {
            this->setJLowNeumann(gcm);
        }
        // jHi:
        if (BC.jHi.type == BCTag::Dirichlet) {
            this->setJHighDirichlet(BC.jHi.value);
        } else if (BC.jHi.type == BCTag::Neumann) {
            this->setJHighNeumann(gcm);
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
    
    inline void setILowNeumann(int ghost_cell_margin) {
        parallel_for(Range<int>(0, gcm), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
            this->operator()(i, j) = this->operator()(gcm, j);
        });
    }
    
    inline void setIHighNeumann(int ghost_cell_margin) {
        parallel_for(Range<int>(ni - gcm, ni), Range<int>(gcm, nj - gcm), [&] (int i, int j) {
            this->operator()(i, j) = this->operator()(ni - gcm - 1, j);
        });
    }
    
    inline void setJLowNeumann(int ghost_cell_margin) {
        parallel_for(Range<int>(gcm, ni - gcm), Range<int>(0, gcm), [&] (int i, int j) {
            this->operator()(i, j) = this->operator()(i, gcm);
        });
    }
    
    inline void setJHighNeumann(int ghost_cell_margin) {
        parallel_for(Range<int>(gcm, ni - gcm), Range<int>(nj - gcm, nj), [&] (int i, int j) {
            this->operator()(i, j) = this->operator()(i, nj - gcm - 1);
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
