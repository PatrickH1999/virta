#include <chrono>
#include <cmath>
#include <iostream>

#include "src/BCStruct.H"
#include "src/Field2D.H"
#include "src/Functions.H"
#include "src/Prob.H"

using DefaultScheme = virta::Central;
using Real = double;
using Field = virta::Field2D<Real>;

constexpr virta::BCTag Neumann   = virta::BCTag::Neumann;
constexpr virta::BCTag Dirichlet = virta::BCTag::Dirichlet;

int main() {
    constexpr Real PI = 3.14159265358979323846264338327950;
    constexpr int N = 128;
    constexpr Real dx = (16 * PI) / N;
    constexpr int max_step = 5000;
    constexpr Real dt = 0.00001;
    constexpr int gcm = 3;

    constexpr Real g = 9.81;
   
    virta::Prob<Real, Field> prob(N, N);

    virta::BCStruct<Real, 2> BC = {{Neumann, 0.0}, {Neumann, 0.0}, {Neumann, 0.0}, {Neumann, 0.0}};
    
    Field& h = prob.add(0.0, BC, gcm, false, false);
    Field& u = prob.add(0.0, BC, gcm, false, false);
    Field& v = prob.add(0.0, BC, gcm, false, false);
   
    Field& hu = prob.add(0.0, BC, gcm, false, false);
    Field& hv = prob.add(0.0, BC, gcm, false, false);
    Field& huu = prob.add(0.0, BC, gcm, false, false);
    Field& huv = prob.add(0.0, BC, gcm, false, false);
    Field& hvv = prob.add(0.0, BC, gcm, false, false);

    Field& dhu_dx = prob.add(0.0, BC, gcm, false, false);
    Field& dhv_dy = prob.add(0.0, BC, gcm, false, false);
    Field& dhuu_dx = prob.add(0.0, BC, gcm, false, false);
    Field& dhuv_dy = prob.add(0.0, BC, gcm, false, false);
    Field& dhuv_dx = prob.add(0.0, BC, gcm, false, false);
    Field& dhvv_dy = prob.add(0.0, BC, gcm, false, false);

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
            prob.setBC();

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
