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

template <int dim, typename Field2D, typename Real>
inline Real grad_central(const Field2D& base, const int i, const int j, const Real dx) {
    constexpr int iFac = (dim == 0) ? 1 : 0;
    constexpr int jFac = (dim == 1) ? 1 : 0;
    return (base(i + iFac, j + jFac) - base(i - iFac, j - jFac)) / (2 * dx);
}

template <int dim, typename Field2D, typename Real>
inline Real grad_central_stag(const Field2D& base, const int i, const int j, const Real dx, const int stagDeriv) {
    constexpr int iFac = (dim == 0) ? 1 : 0;
    constexpr int jFac = (dim == 1) ? 1 : 0;

    bool stagEq = (base.stag == stagDeriv);
    bool stagBaseAligned = (base.stag == dim);
    bool stagDerivAligned = (stagDeriv == dim);
    bool noStagBase = (base.stag == -1);
    bool noStagDeriv = (stagDeriv == -1);
    
    if (stagBaseAligned && noStagDeriv) {
        return (base(i + iFac, j + jFac) - base(i, j)) / dx;
    } else if (noStagBase && stagDerivAligned) {
        return (base(i, j) - base(i - iFac, j - jFac)) / dx;
    } else if (!stagBaseAligned && noStagDeriv) {
        return (base(i + iFac + jFac, j + iFac + jFac) - base(i - iFac + jFac, j + iFac - jFac) + base(i + iFac, j + jFac) - base(i - iFac, j - jFac)) / (4 * dx);
    } else if (noStagBase && !stagDerivAligned) {
        return (base(i + iFac, j + jFac) - base(i - iFac, j - jFac) + base(i + iFac - jFac, j - iFac + jFac) - base(i - iFac - jFac, j - iFac - jFac)) / (4 * dx);
    } else if (stagBaseAligned && !stagDerivAligned) {
        return (base(i + iFac, j + jFac) - base(i, j) + base(i + iFac - jFac, j - iFac + jFac) - base(i - jFac, j - iFac)) / (2 * dx);
    } else if (!stagBaseAligned && stagDerivAligned) {
        return (base(i + jFac, j + iFac) - base(i - iFac + jFac, j + iFac - jFac) + base(i, j) - base(i - iFac, j - jFac)) / (2 * dx);
    } else {
        throw std::logic_error("Invalid staggering.");
        return 0.0;
    }
}

template <int dim, typename Field2D, typename Real>
inline Real grad_central4(const Field2D& base, const int i, const int j, const Real dx) {
    constexpr int iFac = (dim == 0) ? 1 : 0;
    constexpr int jFac = (dim == 1) ? 1 : 0;
    
    return (-base(i + 2*iFac, j + 2*jFac) + 8 * base(i + iFac, j + jFac) - 8 * base(i - iFac, j - jFac) + base(i - 2*iFac, j - 2*jFac)) / (12 * dx);
}

template <int dim, typename Field2D, typename Real>
inline Real grad_central6(const Field2D& base, const int i, const int j, const Real dx) {
    constexpr int iFac = (dim == 0) ? 1 : 0;
    constexpr int jFac = (dim == 1) ? 1 : 0;
    
    return (-base(i + 3*iFac, j + 3*jFac) + 9 * base(i + 2*iFac, j + 2*jFac) - 45 * base(i + iFac, j + jFac) + 45 * base(i - iFac, j - jFac) - 9 * base(i - 2*iFac, j - 2*jFac) + base(i - 3*iFac, j - 3*jFac)) / (60 * dx);
}

template <int dim, typename Field2D, typename Real>
inline Real grad_upwind(const Field2D& base, const int i, const int j, const Real dx, const Field2D& u) {
    constexpr int iFac = (dim == 0) ? 1 : 0;
    constexpr int jFac = (dim == 1) ? 1 : 0;
    
    if (u(i, j) > 0) {
        return (base(i, j) - base(i - iFac, j - jFac)) / dx;
    } else if (u(i, j) < 0) {
        return (base(i + iFac, j + jFac) - base(i, j)) / dx;
    } else {
        return (base(i + iFac, j + jFac) - base(i - iFac, j - jFac)) / (2 * dx);
    }
}

template <int dim, typename Field2D, typename Real>
inline Real grad(GradScheme scheme, const Field2D& base, const int i, const int j, const Real dx, const int stagDeriv, const Field2D& u) {
    bool central = (scheme == GradScheme::Central);
    bool central4 = (scheme == GradScheme::Central4);
    bool central6 = (scheme == GradScheme::Central6);
    bool upwind = (scheme == GradScheme::Upwind);
    bool stagEq = (base.stag == stagDeriv);

    if (central && stagEq) return grad_central<dim>(base, i, j, dx);
    else if (central && !stagEq) return grad_central_stag<dim>(base, i, j, dx, stagDeriv);
    else if (central4 && stagEq) return grad_central4<dim>(base, i, j, dx);
    else if (central6 && stagEq) return grad_central6<dim>(base, i, j, dx);
    else if (upwind && stagEq) return grad_upwind<dim>(base, i, j, dx, u);
    else {
        throw std::logic_error("Invalid grad scheme or grad scheme/staggering combination.");
        return 0.0;
    }
}

template <typename Field2D, typename Real>
inline Real ddx(GradScheme scheme, const Field2D& base, const int i, const int j, const Real dx, const int stagDeriv, const Field2D& u) {
    return grad<0>(scheme, base, i, j, dx, stagDeriv, u);
}

template <typename Field2D, typename Real>
inline Real ddy(GradScheme scheme, const Field2D& base, const int i, const int j, const Real dx, const int stagDeriv, const Field2D& v) {
    return grad<1>(scheme, base, i, j, dx, stagDeriv, v);
}

} // namespace virta

#endif
