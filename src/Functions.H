#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#ifdef USE_OMP
    #include <omp.h>
#endif

#include "Field2D.H"
#include "Grad.H"
#include "Range.H"

namespace virta {

/**
 * @defgroup Functions Functions
 * @brief High-level (parallel) utilities for virta.
 *
 * This module provides wrappers for one-dimensional and two-dimensional
 * parallel loops, as well as a simple parallel region helper.
 * @{
 */

/**
 * @brief Execute a 1D parallel loop over a Range.
 *
 * If `USE_OMP` is defined, this wraps an `#pragma omp for` loop.
 *
 * @tparam Func Callable with signature `void(int)`.
 * @param R_i   Range specifying [begin, end, step).
 * @param f     Function object or lambda to execute for each index.
 */
template <typename Func>
inline void parallel_for(const Range<int>& R_i, Func&& f) {
#ifdef USE_OMP
    #pragma omp for
#endif
    for (int i = R_i.begin; i < R_i.end; i += R_i.step) {
        f(i);
    }
}

/**
 * @brief Execute a 2D parallel loop over two nested Ranges.
 *
 * Uses OpenMP `collapse(2)` to merge the loop nests when available.
 *
 * @tparam Func Callable with signature `void(int,int)`.
 * @param R_i   Horizontal index range.
 * @param R_j   Vertical index range.
 * @param f     Lambda to execute for each (i,j) pair.
 */
template <typename Func>
inline void parallel_for(const Range<int>& R_i,
                         const Range<int>& R_j,
                         Func&& f)
{
#ifdef USE_OMP
    #pragma omp for collapse(2)
#endif
    for (int j = R_j.begin; j < R_j.end; j += R_j.step) {
        for (int i = R_i.begin; i < R_i.end; i += R_i.step) {
            f(i, j);
        }
    }
}

/**
 * @brief Run a function inside an OpenMP parallel region.
 *
 * @tparam Func Callable to be executed by each thread.
 * @param f     Function/lambda to execute inside the region.
 */
template <typename Func>
inline void parallel_region(Func&& f) {
#ifdef USE_OMP
    #pragma omp parallel
    {
        f();
    }
#else
    f();
#endif
}

template <typename Field2D>
inline void interpolate(const Field2D& orig, Field2D& dest) {
    int iFac = (orig.stag == 0) ? 1 : 0;
    int jFac = (orig.stag == 1) ? 1 : 0;
    bool noStag = (orig.stag == -1);
    if (!noStag) {
        parallel_for(Range<int>(0, dest.ni), Range<int>(0, dest.nj), [&] (int i, int j) {
            dest(i, j) = (orig(i + iFac, j + jFac) + orig(i, j)) / 2;
        });
    }
}

inline int gcm(GradScheme scheme) {
    if (scheme == GradScheme::Central) return 1;
    else if (scheme == GradScheme::Central4) return 2;
    else if (scheme == GradScheme::Central6) return 3;
    else if (scheme == GradScheme::Upwind) return 1;
    else return -1;
}

/** @} */ // end of Functions

} // namespace virta

#endif
