#include <chrono>
#include <cmath>
#include <iostream>

#include "src/BCStruct.H"
#include "src/Field2D.H"
#include "src/Functions.H"
#include "src/Grad.H"
#include "src/Prob.H"

constexpr bool stag = true;
constexpr int hStag = -1;
constexpr int uStag = (stag) ? 0 : -1;
constexpr int vStag = (stag) ? 1 : -1;
constexpr virta::BCTag Neumann   = virta::BCTag::Neumann;
constexpr virta::BCTag Dirichlet = virta::BCTag::Dirichlet;
constexpr virta::GradScheme DefaultGradScheme = virta::GradScheme::Central;

using Real = double;
using Field = virta::Field2D<Real>;

int main() {
    constexpr Real PI = 3.14159265358979323846264338327950;
    constexpr int ni = 256;
    constexpr int nj = 128;
    constexpr Real dx = (32 * PI) / ni;
    constexpr int max_step = 100000;
    constexpr Real dt = 0.00005;
    int gcm = virta::gcm(DefaultGradScheme);

    constexpr Real g = 9.81;
    constexpr Real h_level = 5.0;
   
    virta::Prob<Real, Field> prob(ni, nj);

    constexpr virta::BCStruct<Real, 2> BC = {{Neumann, 0.0}, {Neumann, 0.0}, {Neumann, 0.0}, {Neumann, 0.0}};

    Field& h = prob.add(0.0, BC, gcm, hStag);
    Field& u = prob.add(0.0, BC, gcm, hStag);
    Field& v = prob.add(0.0, BC, gcm, hStag);
   
    Field& hu = prob.add(0.0, BC, gcm, uStag);
    Field& hu_interp = prob.add(0.0, BC, gcm, hStag);
    Field& hv = prob.add(0.0, BC, gcm, vStag);
    Field& hv_interp = prob.add(0.0, BC, gcm, hStag);
    Field& huu = prob.add(0.0, BC, gcm, hStag);
    Field& huv = prob.add(0.0, BC, gcm, hStag);
    Field& hvu = prob.add(0.0, BC, gcm, hStag);
    Field& hvv = prob.add(0.0, BC, gcm, hStag);

    Field& dhu_dx = prob.add(0.0, BC, gcm, hStag);
    Field& dhv_dy = prob.add(0.0, BC, gcm, hStag);
    Field& dhuu_dx = prob.add(0.0, BC, gcm, uStag);
    Field& dhuv_dy = prob.add(0.0, BC, gcm, uStag);
    Field& dhvu_dx = prob.add(0.0, BC, gcm, vStag);
    Field& dhvv_dy = prob.add(0.0, BC, gcm, vStag);

    auto t0 = std::chrono::high_resolution_clock::now();
    virta::parallel_region([&]() {

        // Initialize:
        virta::parallel_for(virta::Range<int>(0, h.ni), virta::Range<int>(0, h.nj), [&](int i, int j) {
            //Real r1 = std::sqrt(std::pow((i - 0.25 * h.ni) * dx, 2) + std::pow((j - 0.5 * h.nj) * dx, 2));
            //Real r2 = std::sqrt(std::pow((i - 0.75 * h.ni) * dx, 2) + std::pow((j - 0.5 * h.nj) * dx, 2));
            Real r = std::sqrt(std::pow((i - 0.5 * h.ni) * dx, 2) + std::pow((j - 0.5 * h.nj) * dx, 2));
            //h(i, j) = 1000 + 5 * std::tanh(-r1 + 0.5 * PI) + 5 * std::tanh(-r2 + 0.5 * PI);
            h(i, j) = h_level + 0.1 * std::tanh(-r + 0.5 * PI);
        });
        
        // Compute:
        for (int n = 0; n < max_step; n++) {
            #pragma omp single
            {
                if (n % 1000 == 0) std::cout << "Time step: " << n << '\n';
            }

            // Boundary conditions:
            prob.setBC();

            // Derivatives:
            virta::ddx(DefaultGradScheme, hu, dhu_dx, dx, u, gcm);
            virta::ddy(DefaultGradScheme, hv, dhv_dy, dx, v, gcm);
            virta::ddx(DefaultGradScheme, huu, dhuu_dx, dx, u, gcm);
            virta::ddy(DefaultGradScheme, huv, dhuv_dy, dx, v, gcm);
            virta::ddx(DefaultGradScheme, hvu, dhvu_dx, dx, u, gcm);
            virta::ddy(DefaultGradScheme, hvv, dhvv_dy, dx, v, gcm);

            // Time advance:
            virta::parallel_for(virta::Range<int>(gcm, hu.ni - gcm), virta::Range<int>(gcm, hu.nj - gcm), [&](int i, int j) {
                hu(i, j) -= dt * (dhuu_dx(i, j) + dhuv_dy(i, j));   // Momentum eq. (x)
            });
            virta::parallel_for(virta::Range<int>(gcm, hv.ni - gcm), virta::Range<int>(gcm, hv.nj - gcm), [&](int i, int j) {
                hv(i, j) -= dt * (dhvu_dx(i, j) + dhvv_dy(i, j));   // Momentum eq. (y)
            });    
            virta::interpolate(hu, hu_interp);
            virta::interpolate(hv, hv_interp);
            virta::parallel_for(virta::Range<int>(gcm, h.ni - gcm), virta::Range<int>(gcm, h.nj - gcm), [&](int i, int j) {
                h(i, j) -= dt * (dhu_dx(i, j) + dhv_dy(i, j));   // Continuity eq.
                
                u(i, j) = hu_interp(i, j) / h(i, j);
                v(i, j) = hv_interp(i, j) / h(i, j);
                huu(i, j) = hu_interp(i, j) * u(i, j) + 0.5 * g * h(i, j) * h(i, j);
                huv(i, j) = hu_interp(i, j) * v(i, j);
                hvu(i, j) = hv_interp(i, j) * u(i, j);
                hvv(i, j) = hv_interp(i, j) * v(i, j) + 0.5 * g * h(i, j) * h(i, j);
            });    
        }
    });
    auto t1 = std::chrono::high_resolution_clock::now();
    h.write_binary("h.virta.shallow_water.plt");
    std::cout << "Time elapsed: " << std::chrono::duration<Real>(t1 - t0).count() << " s" << '\n';
}
