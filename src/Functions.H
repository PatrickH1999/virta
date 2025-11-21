#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <utility>
#ifdef USE_OMP
    #include <omp.h>
#endif

#include "Field.H"
#include "Range.H"

namespace virta {

template <typename Func>
inline void parallel_for(const Range<int>& R_i, Func&& f) {
    #ifdef USE_OMP
        #pragma omp for
    #endif
    for (int i = R_i.begin; i < R_i.end; i += R_i.step) {
        f(i);
    }
}

template <typename Func>
inline void parallel_for(const Range<int>& R_i, const Range<int>& R_j, Func&& f) {
    #ifdef USE_OMP
        #pragma omp for collapse(2)
    #endif
    for (int j = R_j.begin; j < R_j.end; j += R_j.step) {
        for (int i = R_i.begin; i < R_i.end; i += R_i.step) {
            f(i, j);
        }
    }
}

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

template <typename Func, typename Real, typename Derived>
inline void parallel_for_all(const Field<Real, Derived>& field, Func&& func)
{
    parallel_for(Range<int>(0, field.n), std::forward<Func>(func));
}

} // namespace virta

#endif
