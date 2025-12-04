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
    constexpr int max_step = 10000;
    constexpr double dt = 0.0001;

    constexpr double g = 9.81;
    constexpr double baseline = 10.0;
    constexpr double radius = PI / 2;
    constexpr double diff = 1.0;
    
    virta::Field2D<double> h(N, N, 0.0);
    auto h_ = h.view();
    virta::Field2D<double> u(N, N, 0.0);
    auto u_ = u.view();
    virta::Field2D<double> v(N, N, 0.0);
    auto v_ = v.view();
   
    virta::Field2D<double> hu = h;
    auto hu_ = hu.view();
    virta::Field2D<double> hv = h;
    auto hv_ = hv.view();
    virta::Field2D<double> huu = h;
    auto huu_ = huu.view();
    virta::Field2D<double> huv = h;
    auto huv_ = huv.view();
    virta::Field2D<double> hvv = h;
    auto hvv_ = hvv.view();

    virta::Field2D<double> dhu_dx = h;
    auto dhu_dx_ = dhu_dx.view();
    virta::Field2D<double> dhv_dy = h;
    auto dhv_dy_ = dhv_dy.view();
    virta::Field2D<double> dhuu_dx = h;
    auto dhuu_dx_ = dhuu_dx.view();
    virta::Field2D<double> dhuv_dy = h;
    auto dhuv_dy_ = dhuv_dy.view();
    virta::Field2D<double> dhuv_dx = h;
    auto dhuv_dx_ = dhuv_dx.view();
    virta::Field2D<double> dhvv_dy = h;
    auto dhvv_dy_ = dhvv_dy.view();

    auto t0 = std::chrono::high_resolution_clock::now();
    virta::parallel_region([&]() {
        
        // Initialize:
        virta::parallel_for(virta::Range<int>(0, N), virta::Range<int>(0, N), [&](int i, int j) {
            double r = std::sqrt(std::pow((i - 0.5 * N) * dx, 2) + std::pow((j - 0.5 * N) * dx, 2));
            if (r < radius) {
                h_(i, j) = baseline + diff;
            } else {
                h_(i, j) = baseline;
            }

            hu_(i, j) = h_(i, j) * u_(i, j);
            hv_(i, j) = h_(i, j) * v_(i, j);
            huu_(i, j) = h_(i, j) * u_(i, j) * u_(i, j) + 0.5 * g * h_(i, j) * h_(i, j);
            huv_(i, j) = h_(i, j) * u_(i, j) * v_(i, j);
            hvv_(i, j) = h_(i, j) * v_(i, j) * v_(i, j) + 0.5 * g * h_(i, j) * h_(i, j);
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
