#include <fstream>
#include <iostream>
#include <vector>

namespace virta {

struct Range {
    int begin;
    int end;
    int step;

    Range(int begin_, int end_, int step_ = 1) : begin(begin_), end(end_), step(step_) {}
};

template <typename Func>
inline void parallel_for(const Range& R_i, Func&& f)
{
    for (int i = R_i.begin; i < R_i.end; i += R_i.step) {
        f(i);
    }
}

template <typename Func>
inline void parallel_for(const Range& R_i, const Range& R_j, Func&& f)
{
    for (int i = R_i.begin; i < R_i.end; i += R_i.step) {
        for (int j = R_j.begin; j < R_j.end; j += R_j.step) {
            f(i, j);
        }
    }
}

template<typename Real, typename Derived>
class Field {

public:
    const std::size_t n;

    explicit Field(std::size_t n_) : n(n_), data_(n_, Real(0)) {}
    explicit Field(std::size_t n_, Real value_) : n(n_), data_(n_, Real(value_)) {}

    virtual ~Field() = default;

    Derived operator+(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        parallel_for_all(result, [&](int i) {
            result.data_[i] += other.data_[i];
        });
        return result;
    }

    Derived operator+(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        parallel_for_all(result, [&](int i) {
            result.data_[i] += other;
        });
        return result;
    }

    Derived operator-(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        parallel_for_all(result, [&](int i) {
            result.data_[i] -= other.data_[i];
        });
        return result;
    }

    Derived operator-(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        parallel_for_all(result, [&](int i) {
            result.data_[i] -= other;
        });
        return result;
    }

    Derived operator-() const {
        Derived result(static_cast<const Derived&>(*this));
        parallel_for_all(result, [&](int i) {
            result.data_[i] = -result.data_[i];
        });
        return result;
    }

    Derived operator*(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        parallel_for_all(result, [&](int i) {
            result.data_[i] *= other.data_[i];
        });
        return result;
    }

    Derived operator*(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        parallel_for_all(result, [&](int i) {
            result.data_[i] *= other;
        });
        return result;
    }

    Derived operator/(const Derived& other) const {
        Derived result(static_cast<const Derived&>(*this));
        parallel_for_all(result, [&](int i) {
            result.data_[i] /= other.data_[i];
        });
        return result;
    }

    Derived operator/(Real other) const {
        Derived result(static_cast<const Derived&>(*this));
        parallel_for_all(result, [&](int i) {
            result.data_[i] /= other;
        });
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

    void write_binary(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&this->ni), sizeof(this->ni));
        out.write(reinterpret_cast<const char*>(&this->nj), sizeof(this->nj));
        out.write(reinterpret_cast<const char*>(this->data_.data()), this->data_.size() * sizeof(Real));
    }

};

template <typename Func, typename Real, typename Derived>
inline void parallel_for_all(const Field<Real, Derived>& field, Func&& func)
{
    parallel_for(Range(0, field.n), std::forward<Func>(func));
}

} // namespace virta

int main() {
    std::cout << "Hello, World!" << '\n';
    virta::Field2D<double> f1(100, 100, 1.0); 
    virta::Field2D<double> f2 = f1 / 7.2315;
    f2.write_binary("output.bin");
}
