#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cassert>
#ifdef USE_OMP
    #include <omp.h>
#endif

#include "Field.H"
#include "Field2D.H"
#include "Prob.H"
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

template <Staggering stagOrig, typename Field2D>
inline void interpolate(const Field2D& orig, Field2D& dest) {
    constexpr int iFac = (stagOrig == Staggering::iStag) ? 1 : 0;
    constexpr int jFac = (stagOrig == Staggering::jStag) ? 1 : 0;
    constexpr bool noStag = (stagOrig == Staggering::noStag);
    bool sizesOk = ((orig.ni == (dest.ni + iFac)) && (orig.nj == (dest.nj + jFac)));
    //assert(sizesOk && "Orig must be +1 larger in staggering direction than dest.");
    if constexpr (!noStag) {
        parallel_for(Range<int>(0, dest.ni), Range<int>(0, dest.nj), [&] (int i, int j) {
            dest(i, j) = (orig(i + iFac, j + jFac) + orig(i, j)) / 2;
        });
    }
}

/** @} */ // end of Functions

} // namespace virta

#endif
