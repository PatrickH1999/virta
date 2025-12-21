#ifndef GRAD_H
#define GRAD_H

#include <utility>
#ifdef USE_OMP
    #include <omp.h>
#endif

#include "Field.H"
#include "Field2D.H"
#include "Range.H"

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

template <GradScheme Scheme, typename Field2D, typename Real>
inline void ddx(const Field2D& base, Field2D& deriv, Real dx, const Field2D& u, int ghost_cell_margin = 0)
{
    constexpr bool central = (Scheme == GradScheme::Central);
    constexpr bool central4 = (Scheme == GradScheme::Central4);
    constexpr bool central6 = (Scheme == GradScheme::Central6);
    constexpr bool upwind = (Scheme == GradScheme::Upwind);

    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, base.ni - gcm), Range<int>(gcm, base.nj - gcm), [&] (int i, int j) {
        if constexpr (central) {
            deriv(i, j) = (base(i + 1, j) - base(i - 1, j)) / (2 * dx);
        } else if constexpr (central4) {
            deriv(i, j) = (-base(i + 2, j) + 8 * base(i + 1, j) - 8 * base(i - 1, j) + base(i - 2, j)) / (12 * dx);
        } else if constexpr (central6) {
            deriv(i, j) = (-base(i + 3, j) + 9 * base(i + 2, j) - 45 * base(i + 1, j) + 45 * base(i - 1, j) - 9 * base(i - 2, j) + base(i - 3, j)) / (60 * dx);
        } else if constexpr (upwind) {
            if (u(i, j) > 0) {
                deriv(i, j) = (base(i, j) - base(i - 1, j)) / dx;
            } else if (u(i, j) < 0) {
                deriv(i, j) = (base(i + 1, j) - base(i, j)) / dx;
            } else {
                deriv(i, j) = (base(i + 1, j) - base(i - 1, j)) / (2 * dx);
            }
        } else {
            static_assert((Scheme != Scheme), "Unknown finite-difference scheme in ddx.");
        }
    });
}

template <GradScheme Scheme, typename Field2D, typename Real>
inline void ddy(const Field2D& base, Field2D& deriv, Real dy, const Field2D& v, int ghost_cell_margin = 0)
{
    constexpr bool central = (Scheme == GradScheme::Central);
    constexpr bool central4 = (Scheme == GradScheme::Central4);
    constexpr bool central6 = (Scheme == GradScheme::Central6);
    constexpr bool upwind = (Scheme == GradScheme::Upwind);

    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, base.ni - gcm), Range<int>(gcm, base.nj - gcm), [&] (int i, int j) {
        if constexpr (central) {
            deriv(i, j) = (base(i, j + 1) - base(i, j - 1)) / (2 * dy);
        } else if constexpr (central4) {
            deriv(i, j) = (-base(i, j + 2) + 8 * base(i, j + 1) - 8 * base(i, j - 1) + base(i, j - 2)) / (12 * dy);
        } else if constexpr (central6) {
            deriv(i, j) = (-base(i, j + 3) + 9 * base(i, j + 2) - 45 * base(i, j + 1) + 45 * base(i, j - 1) - 9 * base(i, j - 2) + base(i, j - 3)) / (60 * dy);
        } else if constexpr (upwind) {
            if (v(i, j) > 0) {
                deriv(i, j) = (base(i, j) - base(i, j - 1)) / dy;
            } else if (v(i, j) < 0) {
                deriv(i, j) = (base(i, j + 1) - base(i, j)) / dy;
            } else {
                deriv(i, j) = (base(i, j + 1) - base(i, j - 1)) / (2 * dy);
            }
        } else {
            static_assert((Scheme != Scheme), "Unknown finite-difference scheme in ddy.");
        }
    });
}

/** @} */ // end of Grad

} // namespace virta

#endif
