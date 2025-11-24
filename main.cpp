#include <chrono>
#include <cmath>
#include <iostream>

#include "src/Functions.H"
#include "src/Field2D.H"

int main() {
    constexpr double PI = 3.14159265358979323846264338327950;
    constexpr int N = 1000;
    double dx = (16 * PI) / N;
    virta::Field2D<double> base(N, N, 0.0);
    virta::Field2D<double> deriv = base;

    auto t0 = std::chrono::high_resolution_clock::now();
    virta::parallel_region([&]() {
        for (int n = 0; n < 1000; n++) {
            virta::parallel_for(virta::Range<int>(0, N), virta::Range<int>(0, N), [&](int i, int j) {
                double x = (-0.5 * N + i) * dx;
                double y = (-0.5 * N + j) * dx;
                base(i, j) = std::sin(x) * std::sin(y);
            });
        }
        virta::ddx(base, deriv, dx, 1);
    });
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Time elapsed: " << std::chrono::duration<double>(t1 - t0).count() << " s" << '\n';
    base.write_binary("base.virta.plt");
    deriv.write_binary("deriv.virta.plt");
}
