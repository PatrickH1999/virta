#include <chrono>
#include <cmath>
#include <iostream>

#include "src/Functions.H"
#include "src/Field2D.H"

using DefaultScheme = virta::Central;

int main() {
    constexpr double PI = 3.14159265358979323846264338327950;
    constexpr int N = 256;
    constexpr double dx = (16 * PI) / N;
    constexpr int max_step = 100000;
    constexpr double dt = 0.00001;

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
            //double r = std::abs(i * dx - 0.5 * N * dx);
            h(i, j) = 1000 + 5 * std::tanh(-r + 0.5 * PI);

            hu(i, j) = h(i, j) * u(i, j);
            hv(i, j) = h(i, j) * v(i, j);
            huu(i, j) = h(i, j) * u(i, j) * u(i, j) + 0.5 * g * h(i, j) * h(i, j);
            huv(i, j) = h(i, j) * u(i, j) * v(i, j);
            hvv(i, j) = h(i, j) * v(i, j) * v(i, j) + 0.5 * g * h(i, j) * h(i, j);
        });
        
        // Compute:
        for (int n = 0; n < max_step; n++) {
            // Boundary conditions:
            virta::setBoundaryILowNeumann(h, 1); 
            virta::setBoundaryIHighNeumann(h, 1); 
            virta::setBoundaryJLowNeumann(h, 1); 
            virta::setBoundaryJHighNeumann(h, 1); 
            virta::setBoundaryILowNeumann(u, 1); 
            virta::setBoundaryIHighNeumann(u, 1); 
            virta::setBoundaryJLowNeumann(u, 1); 
            virta::setBoundaryJHighNeumann(u, 1); 
            virta::setBoundaryILowNeumann(v, 1); 
            virta::setBoundaryIHighNeumann(v, 1); 
            virta::setBoundaryJLowNeumann(v, 1); 
            virta::setBoundaryJHighNeumann(v, 1); 

            // Derivatives:
            virta::ddx<DefaultScheme>(hu, dhu_dx, dx, u, 1);
            virta::ddy<DefaultScheme>(hv, dhv_dy, dx, v, 1);
            virta::ddx<DefaultScheme>(huu, dhuu_dx, dx, u, 1);
            virta::ddy<DefaultScheme>(huv, dhuv_dy, dx, v, 1);
            virta::ddx<DefaultScheme>(huv, dhuv_dx, dx, u, 1);
            virta::ddy<DefaultScheme>(hvv, dhvv_dy, dx, v, 1);

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
