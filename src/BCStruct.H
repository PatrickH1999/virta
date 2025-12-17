#ifndef BCSTRUCT_H
#define BCSTRUCT_H

namespace virta {

enum class BCTag {
    Dirichlet,
    Neumann
};

template <typename Real>
struct BCEntry {
    BCTag type;
    Real value;
};

template <typename Real, int ndim>
struct BCStruct;

template <typename Real>
struct BCStruct<Real, 1> {
    BCEntry<Real> iLo;
    BCEntry<Real> iHi;
};

template <typename Real>
struct BCStruct<Real, 2> {
    BCEntry<Real> iLo;
    BCEntry<Real> iHi;
    BCEntry<Real> jLo;
    BCEntry<Real> jHi;
};

/** @} */ // end of BC

} // namespace virta

# endif
