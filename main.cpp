#include <iostream>
#include <vector>

namespace virta {

struct Range1D {
    long begin_i;
    long end_i;
    long step_i;

    Range1D(long begin_i_, long end_i_) : begin_i(begin_i_), end_i(end_i_), step_i(1) {}
    Range1D(long begin_i_, long end_i_, long step_i_ = 1) : begin_i(begin_i_), end_i(end_i_), step_i(step_i_) {}
};

struct Range2D {
    long begin_i;
    long end_i;
    long step_i;
    long begin_j;
    long end_j;
    long step_j;

    Range2D(long begin_i_, 
            long end_i_, 
            long begin_j_,
            long end_j_) : begin_i(begin_i_), end_i(end_i_), step_i(1), begin_j(begin_j_), end_j(end_j_), step_j(1) {}
    Range2D(long begin_i_,
            long end_i_,
            long step_i_,
            long begin_j_,
            long end_j_,
            long step_j_) : begin_i(begin_i_), end_i(end_i_), step_i(step_i_), begin_j(begin_j_), end_j(end_j_), step_j(step_j_) {}
};

template<typename Real, typename Derived>
class Field {

public:
    const std::size_t n;

    explicit Field(std::size_t n_) : n(n_), data_(n_, Real(0)) {}
    explicit Field(std::size_t n_, Real value_) : n(n_), data_(n_, Real(value_)) {}

    virtual ~Field() = default;

    Derived operator+(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data_[i] += other.data_[i];
        }
        return result;
    }

    Derived operator+(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data_[i] += other;
        }
        return result;
    }

    Derived operator-(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data_[i] -= other.data_[i];
        }
        return result;
    }

    Derived operator-(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data_[i] -= other;
        }
        return result;
    }

    Derived operator-() const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data_[i] = -result.data_[i];
        }
        return result;
    }

    Derived operator*(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data_[i] *= other.data_[i];
        }
        return result;
    }

    Derived operator*(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data_[i] *= other;
        }
        return result;
    }

    Derived operator/(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data_[i] /= other.data_[i];
        }
        return result;
    }

    Derived operator/(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        for (std::size_t i = 0; i < n; ++i) {
            result.data_[i] /= other;
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
    Field1D(std::size_t ni_, Real value_) : Field<Real, Field1D<Real>>(ni_, value_), ni(ni_) {}

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
    Field2D(std::size_t ni_, std::size_t nj_, Real value_) : Field<Real, Field2D<Real>>(ni_ * nj_, value_), ni(ni_), nj(nj_) {}
    
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
    virta::Field2D<double> f1(10000, 30000, 1.0); 
    virta::Field2D<double> f2 = f1 + f1;
}
