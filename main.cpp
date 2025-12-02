#include <chrono>
#include <cmath>
#include <iostream>

#include "src/Functions.H"
#include "src/Field2D.H"

int main() {
    constexpr double PI = 3.14159265358979323846264338327950;
    constexpr int N = 1000;
    constexpr double dx = (16 * PI) / N;
    constexpr int max_step = 500;
    constexpr double dt = 0.001;

    constexpr double g = 9.81;
    
    virta::Field2D<double> h(N, N, 0.0);
    virta::Field2D<double> u(N, N, 0.0);
    virta::Field2D<double> v(N, N, 0.0);
   
    virta::Field2D<double> hu = h;
    virta::Field2D<double> hv = h;
    virta::Field2D<double> huu = h;
    virta::Field2D<double> huv = h;
    virta::Field2D<double> hvv = h;

    virta::Field2D<double> dhu_dx = h;
    virta::Field2D<double> dhv_dy = h;
    virta::Field2D<double> dhuu_dx = h;
    virta::Field2D<double> dhuv_dy = h;
    virta::Field2D<double> dhuv_dx = h;
    virta::Field2D<double> dhvv_dy = h;

    auto t0 = std::chrono::high_resolution_clock::now();
    virta::parallel_region([&]() {
        
        // Initialize:
        virta::parallel_for(virta::Range<int>(0, N), virta::Range<int>(0, N), [&](int i, int j) {
            double r = std::sqrt(std::pow((i - 0.5 * N) * dx, 2) + std::pow((j - 0.5 * N) * dx, 2));
            h(i, j) = 0.5 + 0.5 * std::tanh(-r + 0.1);

            hu(i, j) = h(i, j) * u(i, j);
            hv(i, j) = h(i, j) * v(i, j);
            huu(i, j) = h(i, j) * u(i, j) * u(i, j) + 0.5 * g * h(i, j) * h(i, j);
            huv(i, j) = h(i, j) * u(i, j) * v(i, j);
            hvv(i, j) = h(i, j) * v(i, j) * v(i, j) + 0.5 * g * h(i, j) * h(i, j);
        });
        
        // Compute:
        for (int n = 0; n < max_step; n++) {
            // Derivatives:
            virta::ddx(hu, dhu_dx, dx, 1);
            virta::ddy(hv, dhv_dy, dx, 1);
            virta::ddx(huu, dhuu_dx, dx, 1);
            virta::ddy(huv, dhuv_dy, dx, 1);
            virta::ddx(huv, dhuv_dx, dx, 1);
            virta::ddy(hvv, dhvv_dy, dx, 1);

            // Time advance:
            virta::parallel_for(virta::Range<int>(0, N), virta::Range<int>(0, N), [&](int i, int j) {
                h(i, j) -= dt * (dhu_dx(i, j) + dhv_dy(i, j));   // Continuity eq.
                hu(i, j) -= dt * (dhuu_dx(i, j) + dhuv_dy(i, j));   // Momentum eq. (x)
                hv(i, j) -= dt * (dhuv_dx(i, j) + dhvv_dy(i, j));   // Momentum eq. (y)
                
                u(i, j) = hu(i, j) / h(i, j);
                v(i, j) = hv(i, j) / h(i, j);
                huu(i, j) = hu(i, j) * u(i, j) + 0.5 * g * h(i, j) * h(i, j);
                huv(i, j) = hu(i, j) * v(i, j);
                hvv(i, j) = hv(i, j) * v(i, j) + 0.5 * g * h(i, j) * h(i, j);
        });    
        }
    });
    auto t1 = std::chrono::high_resolution_clock::now();
    h.write_binary("h.virta.shallow_water.plt");
    std::cout << "Time elapsed: " << std::chrono::duration<double>(t1 - t0).count() << " s" << '\n';
}
