#include <iostream>
#include <vector>

namespace virta {

template<typename Real, typename Derived>
class Field {

public:
    const std::size_t n;

    explicit Field(std::size_t n_) : n(n_), data_(n_, Real(0)) {}

    virtual ~Field() = default;

    Derived operator+(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i] += other.data[i];
        }
        return result;
    }

    Derived operator+(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i] += other;
        }
        return result;
    }

    Derived operator-(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i] -= other.data[i];
        }
        return result;
    }

    Derived operator-(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i] -= other;
        }
        return result;
    }

    Derived operator-() const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i] = -result.data[i];
        }
        return result;
    }

    Derived operator*(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i] *= other.data[i];
        }
        return result;
    }

    Derived operator*(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i] *= other;
        }
        return result;
    }

    Derived operator/(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i] /= other.data[i];
        }
        return result;
    }

    Derived operator/(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data[i] /= other;
        }
        return result;
    }

protected:
    std::vector<Real> data_;

};

template<typename Real>
class Field1D : public Field<Real, Field1D<Real>> {

public:
    static constexpr int ndim = 1;
    const std::size_t ni;

    Field1D(std::size_t ni_) : Field<Real, Field1D<Real>>(ni_), ni(ni_) {}

    Real& operator()(std::size_t i) {
        return this->data[i];
    }

    const Real& operator()(std::size_t i) const {
        return this->data[i];
    }

};

template<typename Real>
class Field2D : public Field<Real, Field2D<Real>> {

public:
    static constexpr int ndim = 2;
    const std::size_t ni;
    const std::size_t nj;

    Field2D(std::size_t ni_, std::size_t nj_) : Field<Real, Field2D<Real>>(ni_ * nj_), ni(ni_), nj(nj_) {}
    
    Real& operator()(std::size_t i, std::size_t j) {
        return this->data[j * ni + i];
    }

    const Real& operator()(std::size_t i, std::size_t j) const {
        return this->data[j * ni + i];
    }

};

} // namespace virta

int main() {
    std::cout << "Hello, World!" << '\n';
    virta::Field1D<float> f1(100); 
    std::cout << "f1 created. ndim = " << f1.ndim << ", n = " << f1.n << '\n';
    virta::Field2D<float> f2(100, 200); 
    std::cout << "f2 created. ndim = " << f2.ndim << ", n = " << f2.n << '\n';
}
