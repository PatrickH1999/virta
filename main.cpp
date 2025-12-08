#include <chrono>
#include <cmath>
#include <iostream>

#include "src/Functions.H"
#include "src/BC.H"
#include "src/Field2D.H"

using DefaultScheme = virta::Central6;

int main() {
    constexpr double PI = 3.14159265358979323846264338327950;
    constexpr int N = 512;
    constexpr double dx = (16 * PI) / N;
    constexpr int max_step = 30000;
    constexpr double dt = 0.00001;
    constexpr int gcm = 3;

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
            double r1 = std::sqrt(std::pow((i - 0.25 * N) * dx, 2) + std::pow((j - 0.5 * N) * dx, 2));
            double r2 = std::sqrt(std::pow((i - 0.75 * N) * dx, 2) + std::pow((j - 0.5 * N) * dx, 2));
            //double r = std::abs(i * dx - 0.5 * N * dx);
            h(i, j) = 1000 + 5 * std::tanh(-r1 + 0.5 * PI) + 5 * std::tanh(-r2 + 0.5 * PI);

            hu(i, j) = h(i, j) * u(i, j);
            hv(i, j) = h(i, j) * v(i, j);
            huu(i, j) = h(i, j) * u(i, j) * u(i, j) + 0.5 * g * h(i, j) * h(i, j);
            huv(i, j) = h(i, j) * u(i, j) * v(i, j);
            hvv(i, j) = h(i, j) * v(i, j) * v(i, j) + 0.5 * g * h(i, j) * h(i, j);
        });
        
        // Compute:
        for (int n = 0; n < max_step; n++) {
            // Boundary conditions:
            virta::BC::setILowNeumann(h, gcm); 
            virta::BC::setIHighNeumann(h, gcm); 
            virta::BC::setJLowNeumann(h, gcm); 
            virta::BC::setJHighNeumann(h, gcm); 
            virta::BC::setILowNeumann(u, gcm); 
            virta::BC::setIHighNeumann(u, gcm); 
            virta::BC::setJLowNeumann(u, gcm); 
            virta::BC::setJHighNeumann(u, gcm); 
            virta::BC::setILowNeumann(v, gcm); 
            virta::BC::setIHighNeumann(v, gcm); 
            virta::BC::setJLowNeumann(v, gcm); 
            virta::BC::setJHighNeumann(v, gcm); 

            // Derivatives:
            virta::ddx<DefaultScheme>(hu, dhu_dx, dx, u, gcm);
            virta::ddy<DefaultScheme>(hv, dhv_dy, dx, v, gcm);
            virta::ddx<DefaultScheme>(huu, dhuu_dx, dx, u, gcm);
            virta::ddy<DefaultScheme>(huv, dhuv_dy, dx, v, gcm);
            virta::ddx<DefaultScheme>(huv, dhuv_dx, dx, u, gcm);
            virta::ddy<DefaultScheme>(hvv, dhvv_dy, dx, v, gcm);

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
