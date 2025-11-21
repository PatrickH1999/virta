#ifndef FIELD1D_H
#define FIELD1D_H

#include "Field.H"

namespace virta {

template<typename Real>
class Field1D : public Field<Real, Field1D<Real>> {

public:
    size_t ni;
    static constexpr int ndim = 1;

    Field1D(size_t ni_) : Field<Real, Field1D<Real>>(ni_), ni(ni_) {}
    Field1D(size_t ni_, Real value_) : Field<Real, Field1D<Real>>(ni_, value_), ni(ni_) {}

    inline Real& operator()(size_t i) {
        return this->data[i];
    }

    inline const Real& operator()(size_t i) const {
        return this->data[i];
    }

};

} // namespace virta

#endif
