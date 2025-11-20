#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#ifdef USE_OMP
    #include <omp.h>
#endif

#include "Functions.H"
#include "Field2D.H"

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
