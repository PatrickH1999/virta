#ifndef GRAD_H
#define GRAD_H

#include <utility>
#ifdef USE_OMP
    #include <omp.h>
#endif

#include "Field2D.H"
#include "Range.H"
#include "Prob.H"

namespace virta {

/**
 * @defgroup Grad Grad
 * @brief High-level (parallel) utilities for virta.
 *
 * This module provides wrappers for one-dimensional and two-dimensional
 * parallel loops, as well as a simple parallel region helper.
 * @{
 */

enum class GradScheme {
    Central,
    Central4,
    Central6,
    Upwind
};

template <int dim, GradScheme scheme, Staggering stagBase = Staggering::noStag, Staggering stagDeriv = Staggering::noStag, typename Field2D, typename Real>
inline void grad(const Field2D& base, Field2D& deriv, Real dx, const Field2D& u, int ghost_cell_margin = 0)
{
    constexpr int iFac = (dim == 0) ? 1 : 0;
    constexpr int jFac = (dim == 1) ? 1 : 0;
    
    constexpr bool central = (scheme == GradScheme::Central);
    constexpr bool central4 = (scheme == GradScheme::Central4);
    constexpr bool central6 = (scheme == GradScheme::Central6);
    constexpr bool upwind = (scheme == GradScheme::Upwind);

    constexpr bool stagEq = (stagBase == stagDeriv);
    constexpr bool stagBaseAligned = (((dim == 0) && (stagBase == Staggering::iStag)) || ((dim == 1) && (stagBase == Staggering::jStag)));
    constexpr bool stagDerivAligned = (((dim == 0) && (stagDeriv == Staggering::iStag)) || ((dim == 1) && (stagDeriv == Staggering::jStag)));
    constexpr bool noStagBase = (stagBase == Staggering::noStag);
    constexpr bool noStagDeriv = (stagDeriv == Staggering::noStag);
 
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, base.ni - gcm), Range<int>(gcm, base.nj - gcm), [&] (int i, int j) {
        if constexpr (central && stagEq) {
            deriv(i, j) = (base(i + iFac, j + jFac) - base(i - iFac, j - jFac)) / (2 * dx);
        } else if constexpr (central && stagBaseAligned && noStagDeriv) {
            deriv(i, j) = (base(i + iFac, j + jFac) - base(i, j)) / dx;
        } else if constexpr (central && noStagBase && stagDerivAligned) {
            deriv(i, j) = (base(i, j) - base(i - iFac, j - jFac)) / dx;
        } else if constexpr (central && !stagBaseAligned && noStagDeriv) {
            deriv(i, j) = (base(i + iFac + jFac, j + iFac + jFac) - base(i - iFac + jFac, j + iFac - jFac) + base(i + iFac, j + jFac) - base(i - iFac, j - jFac)) / (4 * dx);
        } else if constexpr (central && noStagBase && !stagDerivAligned) {
            deriv(i, j) = (base(i + iFac, j + jFac) - base(i - iFac, j - jFac) + base(i + iFac - jFac, j - iFac + jFac) - base(i - iFac - jFac, j - iFac - jFac)) / (4 * dx);
        } else if constexpr (central && stagBaseAligned && !stagDerivAligned) {
            deriv(i, j) = (base(i + iFac, j + jFac) - base(i, j) + base(i + iFac - jFac, j - iFac + jFac) - base(i - jFac, j - iFac)) / (2 * dx);
        } else if constexpr (central && !stagBaseAligned && stagDerivAligned) {
            deriv(i, j) = (base(i + jFac, j + iFac) - base(i - iFac + jFac, j + iFac - jFac) + base(i, j) - base(i - iFac, j - jFac)) / (2 * dx);
        } else if constexpr (central4 && stagEq) {
            deriv(i, j) = (-base(i + 2*iFac, j + 2*jFac) + 8 * base(i + iFac, j + jFac) - 8 * base(i - iFac, j - jFac) + base(i - 2*iFac, j - 2*jFac)) / (12 * dx);
        } else if constexpr (central6 && stagEq) {
            deriv(i, j) = (-base(i + 3*iFac, j + 3*jFac) + 9 * base(i + 2*iFac, j + 2*jFac) - 45 * base(i + iFac, j + jFac) + 45 * base(i - iFac, j - jFac) - 9 * base(i - 2*iFac, j - 2*jFac) + base(i - 3*iFac, j - 3*jFac)) / (60 * dx);
        } else if constexpr (upwind && stagEq) {
            if (u(i, j) > 0) {
                deriv(i, j) = (base(i, j) - base(i - iFac, j - jFac)) / dx;
            } else if (u(i, j) < 0) {
                deriv(i, j) = (base(i + iFac, j + jFac) - base(i, j)) / dx;
            } else {
                deriv(i, j) = (base(i + iFac, j + jFac) - base(i - iFac, j - jFac)) / (2 * dx);
            }
        } else {
            static_assert((scheme != scheme), "Unknown grad scheme or grad scheme/staggering combination.");
        }
    });
}

template <GradScheme scheme, Staggering stagBase = Staggering::noStag, Staggering stagDeriv = Staggering::noStag, typename Field2D, typename Real>
inline void ddx(const Field2D& base, Field2D& deriv, Real dx, const Field2D& u, int ghost_cell_margin = 0) {
    grad<0, scheme, stagBase, stagDeriv>(base, deriv, dx, u, ghost_cell_margin);
}

template <GradScheme scheme, Staggering stagBase = Staggering::noStag, Staggering stagDeriv = Staggering::noStag, typename Field2D, typename Real>
inline void ddy(const Field2D& base, Field2D& deriv, Real dx, const Field2D& u, int ghost_cell_margin = 0) {
    grad<1, scheme, stagBase, stagDeriv>(base, deriv, dx, u, ghost_cell_margin);
}

} // namespace virta

#endif
