#include <iostream>
#include <vector>

template<typename Real>
class Field {

public:
    const std::size_t n;

    explicit Field(std::size_t n_) : n(n_), data_(n_, Real(0)) {}

    virtual ~Field() = default;

protected:
    std::vector<Real> data_;

};

template<typename Real>
class Field1D : public Field<Real> {

public:
    static constexpr int ndim = 1;
    const std::size_t ni;

    Field1D(std::size_t ni_) : Field<Real>(ni_), ni(ni_) {}

};

template<typename Real>
class Field2D : public Field<Real> {

public:
    static constexpr int ndim = 2;
    const std::size_t ni;
    const std::size_t nj;

    Field2D(std::size_t ni_, std::size_t nj_) : Field<Real>(ni_ * nj_), ni(ni_), nj(nj_) {}

};

int main() {
    std::cout << "Hello, World!" << '\n';
    Field1D<float> f1(100); 
    std::cout << "f1 created. ndim = " << f1.ndim << ", n = " << f1.n << '\n';
    Field2D<float> f2(100, 200); 
    std::cout << "f2 created. ndim = " << f2.ndim << ", n = " << f2.n << '\n';
}
