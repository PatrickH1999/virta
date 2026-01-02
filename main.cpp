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
    constexpr int ni = 128;
    constexpr int nj = 128;
    constexpr Real dx = (8 * PI) / ni;
    constexpr int max_step = 50000;
    constexpr Real dt = 0.0001;
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

    auto t0 = std::chrono::high_resolution_clock::now();
    virta::parallel_region([&]() {

        // Initialize:
        virta::parallel_for(virta::Range<int>(0, h.ni), virta::Range<int>(0, h.nj), [&](int i, int j) {
            Real xc = (i + 0.5 - 0.5 * h.ni) * dx;
            Real yc = (j + 0.5 - 0.5 * h.nj) * dx;            
            Real r = std::sqrt(xc*xc + yc*yc);
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

            // Time advance:
            virta::parallel_for(virta::Range<int>(gcm, hu.ni - gcm), virta::Range<int>(gcm, hu.nj - gcm), [&](int i, int j) {
                Real dhuu_dx = virta::ddx(DefaultGradScheme, huu, i, j, dx, uStag, u);
                Real dhuv_dy = virta::ddy(DefaultGradScheme, huv, i, j, dx, uStag, v);
                hu(i, j) -= dt * (dhuu_dx + dhuv_dy);   // Momentum eq. (x)
            });
            virta::parallel_for(virta::Range<int>(gcm, hv.ni - gcm), virta::Range<int>(gcm, hv.nj - gcm), [&](int i, int j) {
                Real dhvu_dx = virta::ddx(DefaultGradScheme, hvu, i, j, dx, vStag, u);
                Real dhvv_dy = virta::ddy(DefaultGradScheme, hvv, i, j, dx, vStag, v);
                hv(i, j) -= dt * (dhvu_dx + dhvv_dy);   // Momentum eq. (y)
            });    
            virta::interpolate(hu, hu_interp);
            virta::interpolate(hv, hv_interp);
            virta::parallel_for(virta::Range<int>(gcm, h.ni - gcm), virta::Range<int>(gcm, h.nj - gcm), [&](int i, int j) {
                Real dhu_dx = virta::ddx(DefaultGradScheme, hu, i, j, dx, hStag, u);
                Real dhv_dy = virta::ddy(DefaultGradScheme, hv, i, j, dx, hStag, v);
                h(i, j) -= dt * (dhu_dx + dhv_dy);   // Continuity eq.
                
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
