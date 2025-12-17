#include <chrono>
#include <cmath>
#include <iostream>

#include "src/Functions.H"
#include "src/BC.H"
#include "src/Field2D.H"
#include "src/Prob.H"

using DefaultScheme = virta::Central;
using Real = double;
using Field = virta::Field2D<Real>;

using virta::BCTag::Neumann;
using virta::BCTag::Dirichlet;

int main() {
    constexpr Real PI = 3.14159265358979323846264338327950;
    constexpr int N = 1024;
    constexpr Real dx = (16 * PI) / N;
    constexpr int max_step = 1250;
    constexpr Real dt = 0.00005;
    constexpr int gcm = 3;

    constexpr Real g = 9.81;
   
    virta::BCStruct BC = {Neumann, Neumann, Neumann, Neumann};
    virta::Prob<Real, Field> prob(N, N);
    Field& h = prob.add(BC, 0.0);
    Field& u = prob.add(BC, 0.0);
    Field& v = prob.add(BC, 0.0);
   
    Field& hu = prob.add(BC, 0.0);
    Field& hv = prob.add(BC, 0.0);
    Field& huu = prob.add(BC, 0.0);
    Field& huv = prob.add(BC, 0.0);
    Field& hvv = prob.add(BC, 0.0);

    Field& dhu_dx = prob.add(BC, 0.0);
    Field& dhv_dy = prob.add(BC, 0.0);
    Field& dhuu_dx = prob.add(BC, 0.0);
    Field& dhuv_dy = prob.add(BC, 0.0);
    Field& dhuv_dx = prob.add(BC, 0.0);
    Field& dhvv_dy = prob.add(BC, 0.0);

    auto t0 = std::chrono::high_resolution_clock::now();
    virta::parallel_region([&]() {
        
        // Initialize:
        virta::parallel_for(virta::Range<int>(0, N), virta::Range<int>(0, N), [&](int i, int j) {
            Real r1 = std::sqrt(std::pow((i - 0.25 * N) * dx, 2) + std::pow((j - 0.5 * N) * dx, 2));
            Real r2 = std::sqrt(std::pow((i - 0.75 * N) * dx, 2) + std::pow((j - 0.5 * N) * dx, 2));
            //Real r = std::abs(i * dx - 0.5 * N * dx);
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
            virta::setILowNeumann(h, gcm); 
            virta::setIHighNeumann(h, gcm); 
            virta::setJLowNeumann(h, gcm); 
            virta::setJHighNeumann(h, gcm); 
            virta::setILowNeumann(u, gcm); 
            virta::setIHighNeumann(u, gcm); 
            virta::setJLowNeumann(u, gcm); 
            virta::setJHighNeumann(u, gcm); 
            virta::setILowNeumann(v, gcm); 
            virta::setIHighNeumann(v, gcm); 
            virta::setJLowNeumann(v, gcm); 
            virta::setJHighNeumann(v, gcm); 

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
    std::cout << "Time elapsed: " << std::chrono::duration<Real>(t1 - t0).count() << " s" << '\n';
}
