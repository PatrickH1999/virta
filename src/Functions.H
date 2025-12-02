#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <utility>
#ifdef USE_OMP
    #include <omp.h>
#endif

#include "Field.H"
#include "Field2D.H"
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

/**
 * @brief Convenience wrapper that loops over all entries of a Field.
 *
 * Equivalent to:
 * ```cpp
 * parallel_for(Range<int>(0, field.n), func);
 * ```
 *
 * @tparam Func     Callable with signature `void(int)`.
 * @tparam Real     Floating-point type of the field.
 * @tparam Derived  CRTP derived field type.
 * @param field     Field whose elements should be iterated.
 * @param func      Lambda applied to each index.
 */
template <typename Func, typename Real, typename Derived>
inline void parallel_for_all(const Field<Real, Derived>& field, Func&& func)
{
    parallel_for(Range<int>(0, field.n), std::forward<Func>(func));
}

template <typename Real>
inline void ddx(const Field2D<Real>& base, Field2D<Real>& deriv, Real dx, int ghost_cell_margin = 0)
{
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, base.ni - gcm), Range<int>(gcm, base.nj - gcm), [&] (int i, int j) {
        deriv(i,j) = (base(i + 1, j) - base(i - 1, j)) / (2 * dx);
    });
}

template <typename Real>
inline void ddy(const Field2D<Real>& base, Field2D<Real>& deriv, Real dy, int ghost_cell_margin = 0)
{
    auto& gcm = ghost_cell_margin;
    parallel_for(Range<int>(gcm, base.ni - gcm), Range<int>(gcm, base.nj - gcm), [&] (int i, int j) {
        deriv(i,j) = (base(i, j + 1) - base(i, j - 1)) / (2 * dy);
    });
}

/** @} */ // end of Functions

} // namespace virta

#endif
