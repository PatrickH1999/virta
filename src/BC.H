#ifndef BC_H
#define BC_H

#include <string>

#include "Field1D.H"
#include "Field2D.H"
#include "Range.H"

namespace virta {

enum class BCTag {
    Dirichlet,
    Neumann
};

template <typename FieldXD>
struct BCStruct;

template <typename Real>
struct BCStruct<virta::Field1D<Real>> {
    BCTag iLo;
    BCTag iHi;
};

template <typename Real>
struct BCStruct<virta::Field2D<Real>> {
    BCTag iLo;
    BCTag iHi;
    BCTag jLo;
    BCTag jHi;
};

template <typename Real>
inline void setILowDirichlet(Field2D<Real>& field, Real value, int ghost_cell_margin = 0) {
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(0, gcm), Range<int>(gcm, field.nj - gcm), [&] (int i, int j) {
        field(i, j) = value;
    });
}

template <typename Real>
inline void setIHighDirichlet(Field2D<Real>& field, Real value, int ghost_cell_margin = 0) {
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(field.ni - gcm, field.ni), Range<int>(gcm, field.nj - gcm), [&] (int i, int j) {
        field(i, j) = value;
    });
}

template <typename Real>
inline void setJLowDirichlet(Field2D<Real>& field, Real value, int ghost_cell_margin = 0) {
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, field.ni - gcm), Range<int>(0, gcm), [&] (int i, int j) {
        field(i, j) = value;
    });
}

template <typename Real>
inline void setJHighDirichlet(Field2D<Real>& field, Real value, int ghost_cell_margin = 0) {
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, field.ni - gcm), Range<int>(field.nj - gcm, field.nj), [&] (int i, int j) {
        field(i, j) = value;
    });
}

template <typename Real>
inline void setILowNeumann(Field2D<Real>& field, int ghost_cell_margin = 0) {
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(0, gcm), Range<int>(gcm, field.nj - gcm), [&] (int i, int j) {
        field(i, j) = field(gcm, j);
    });
}

template <typename Real>
inline void setIHighNeumann(Field2D<Real>& field, int ghost_cell_margin = 0) {
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(field.ni - gcm, field.ni), Range<int>(gcm, field.nj - gcm), [&] (int i, int j) {
        field(i, j) = field(field.ni - gcm - 1, j);
    });
}

template <typename Real>
inline void setJLowNeumann(Field2D<Real>& field, int ghost_cell_margin = 0) {
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, field.ni - gcm), Range<int>(0, gcm), [&] (int i, int j) {
        field(i, j) = field(i, gcm);
    });
}

template <typename Real>
inline void setJHighNeumann(Field2D<Real>& field, int ghost_cell_margin = 0) {
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, field.ni - gcm), Range<int>(field.nj - gcm, field.nj), [&] (int i, int j) {
        field(i, j) = field(i, field.nj - gcm - 1);
    });
}

/** @} */ // end of BC

} // namespace virta

# endif
