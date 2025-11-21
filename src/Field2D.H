#ifndef FIELD2D_H
#define FIELD2D_H

#include <cstddef>
#include <fstream>
#include <string>

#include "Field.H"

namespace virta {

template<typename Real>
class Field2D : public Field<Real, Field2D<Real>> {

public:
    size_t ni;
    size_t nj;
    static constexpr int ndim = 2;

    Field2D(size_t ni_, size_t nj_) noexcept : Field<Real, Field2D<Real>>(ni_ * nj_), ni(ni_), nj(nj_) {}
    Field2D(size_t ni_, size_t nj_, Real value_) noexcept : Field<Real, Field2D<Real>>(ni_ * nj_, value_), ni(ni_), nj(nj_) {}
    
    inline Real& operator()(size_t i) noexcept {
        return this->data[i];
    }
    
    inline const Real& operator()(size_t i) const noexcept {
        return this->data[i];
    }
    
    inline Real& operator()(size_t i, size_t j) noexcept {
        return this->data[j * ni + i];
    }

    inline const Real& operator()(size_t i, size_t j) const noexcept {
        return this->data[j * ni + i];
    }

    void write_binary(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&this->ni), sizeof(this->ni));
        out.write(reinterpret_cast<const char*>(&this->nj), sizeof(this->nj));
        out.write(reinterpret_cast<const char*>(this->data.data()), this->data.size() * sizeof(Real));
    }

};

} // namespace virta

#endif
