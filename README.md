## Build & Documentation Instructions

```bash
# Create and enter build directory
mkdir build
cd build

# Configure with OpenMP ON
cmake .. -DUSE_OMP=ON

# Configure with OpenMP OFF
cmake .. -DUSE_OMP=OFF

# Build
make -j

# Build Doxygen documentation
make doc

# Leave build directory
cd ..

# Run
./build/main.ex

# Set number of OpenMP threads
OMP_NUM_THREADS=8 ./build/main.ex

# Docs location
docs/html/index.html
