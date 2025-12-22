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

template <int dim, GradScheme scheme, Staggering stagOrig = Staggering::noStag, Staggering stagDest = Staggering::noStag, typename Field2D, typename Real>
inline void grad(const Field2D& base, Field2D& deriv, Real dx, const Field2D& u, int ghost_cell_margin = 0)
{
    constexpr bool central = (scheme == GradScheme::Central);
    constexpr bool central4 = (scheme == GradScheme::Central4);
    constexpr bool central6 = (scheme == GradScheme::Central6);
    constexpr bool upwind = (scheme == GradScheme::Upwind);
    constexpr bool stagEq = (stagOrig == stagDest);
    constexpr int iFac = (dim == 0) ? 1 : 0;
    constexpr int jFac = (dim == 1) ? 1 : 0;
 
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, base.ni - gcm), Range<int>(gcm, base.nj - gcm), [&] (int i, int j) {
        if constexpr (central && stagEq) {
            deriv(i, j) = (base(i + iFac, j + jFac) - base(i - 1*iFac, j - 1*jFac)) / (2 * dx);
        } else if constexpr (central4 && stagEq) {
            deriv(i, j) = (-base(i + 2*iFac, j + 2*jFac) + 8 * base(i + iFac, j + jFac) - 8 * base(i - 1*iFac, j - 1*jFac) + base(i - 2*iFac, j - 2*jFac)) / (12 * dx);
        } else if constexpr (central6 && stagEq) {
            deriv(i, j) = (-base(i + 3*iFac, j + 3*jFac) + 9 * base(i + 2*iFac, j + 2*jFac) - 45 * base(i + iFac, j + jFac) + 45 * base(i - 1*iFac, j - 1*jFac) - 9 * base(i - 2*iFac, j - 2*jFac) + base(i - 3*iFac, j - 3*jFac)) / (60 * dx);
        } else if constexpr (upwind && stagEq) {
            if (u(i, j) > 0) {
                deriv(i, j) = (base(i, j) - base(i - 1*iFac, j - 1*jFac)) / dx;
            } else if (u(i, j) < 0) {
                deriv(i, j) = (base(i + iFac, j + jFac) - base(i, j)) / dx;
            } else {
                deriv(i, j) = (base(i + iFac, j + jFac) - base(i - 1*iFac, j - 1*jFac)) / (2 * dx);
            }
        } else {
            static_assert((scheme != scheme), "Unknown finite-difference scheme in ddx.");
        }
    });
}

template <GradScheme scheme, Staggering stagOrig = Staggering::noStag, Staggering stagDest = Staggering::noStag, typename Field2D, typename Real>
inline void ddx(const Field2D& base, Field2D& deriv, Real dx, const Field2D& u, int ghost_cell_margin = 0) {
    grad<0, scheme, stagOrig, stagDest>(base, deriv, dx, u, ghost_cell_margin);
}

template <GradScheme scheme, Staggering stagOrig = Staggering::noStag, Staggering stagDest = Staggering::noStag, typename Field2D, typename Real>
inline void ddy(const Field2D& base, Field2D& deriv, Real dx, const Field2D& u, int ghost_cell_margin = 0) {
    grad<1, scheme, stagOrig, stagDest>(base, deriv, dx, u, ghost_cell_margin);
}

} // namespace virta

#endif
