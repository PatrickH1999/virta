#ifndef FIELDVIEW2D_H
#define FIELDVIEW2D_H

#include "Field.H"

namespace virta {

template<typename Real>
struct FieldView2D {
    Real* __restrict__ ptr;
    int ni;

    inline Real& operator()(int i, int j) noexcept {
        return ptr[j*ni + i];
    }
    inline const Real& operator()(int i, int j) const noexcept {
        return ptr[j*ni + i];
    }
};

} // namespace virta

#endif
