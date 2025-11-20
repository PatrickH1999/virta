#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#ifdef USE_OMP
    #include <omp.h>
#endif

namespace virta {

template <typename T>
struct Range {
    T begin;
    T end;
    T step;

    Range(T begin_, T end_, T step_ = 1) : begin(begin_), end(end_), step(step_) {}
};

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

template<typename Real, typename Derived>
class Field {

public:
    size_t n;
    std::vector<Real> data;

    explicit Field(size_t n_) : n(n_), data(n_, Real(0)) {}
    explicit Field(size_t n_, Real value_) : n(n_), data(n_, Real(value_)) {}

    ~Field() = default;
    Field(const Field&) = default;
    Field(Field&&) noexcept = default;
    Field& operator=(const Field&) = default;
    Field& operator=(Field&&) noexcept = default;
};

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

template<typename Real>
class Field2D : public Field<Real, Field2D<Real>> {

public:
    size_t ni;
    size_t nj;
    static constexpr int ndim = 2;

    Field2D(size_t ni_, size_t nj_) : Field<Real, Field2D<Real>>(ni_ * nj_), ni(ni_), nj(nj_) {}
    Field2D(size_t ni_, size_t nj_, Real value_) : Field<Real, Field2D<Real>>(ni_ * nj_, value_), ni(ni_), nj(nj_) {}
    
    inline Real& operator()(size_t i) {
        return this->data[i];
    }
    
    inline const Real& operator()(size_t i) const {
        return this->data[i];
    }
    
    inline Real& operator()(size_t i, size_t j) {
        return this->data[j * ni + i];
    }

    inline const Real& operator()(size_t i, size_t j) const {
        return this->data[j * ni + i];
    }

    void write_binary(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&this->ni), sizeof(this->ni));
        out.write(reinterpret_cast<const char*>(&this->nj), sizeof(this->nj));
        out.write(reinterpret_cast<const char*>(this->data.data()), this->data.size() * sizeof(Real));
    }

};

template <typename Func, typename Real, typename Derived>
inline void parallel_for_all(const Field<Real, Derived>& field, Func&& func)
{
    parallel_for(Range<int>(0, field.n), std::forward<Func>(func));
}

} // namespace virta

int main() {
    virta::Field2D<double> f1(1000, 1000, 215.025);
    virta::Field2D<double> f2(1000, 1000, 0.12513);
    auto t0 = std::chrono::high_resolution_clock::now();
    virta::parallel_region([&]() {
        for (int n=0; n<1000; n++) {
            parallel_for_all(f1, [&](int i) {
                f1(i) = f2(i) * std::pow((f1(i) + f2(i) / (f1(i) - f2(i))), 3.4462) * std::sqrt(f1(i) + f2(i) * f1(i));
            });
        }
    });
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Time elapsed: " << std::chrono::duration<double>(t1 - t0).count() << " s" << '\n';
    //f2.write_binary("output.bin");
}
