cmake_minimum_required(VERSION 3.15)
project(virta LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OMP "Enable OpenMP parallelization" ON)

if (USE_OMP)
    find_package(OpenMP REQUIRED)
    add_compile_definitions(USE_OMP)
endif()

add_executable(main.ex main.cpp)

target_include_directories(main.ex PRIVATE ${PROJECT_SOURCE_DIR}/src)

if (USE_OMP)
    target_link_libraries(main.ex PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_options(main.ex PRIVATE -O3 -march=native)
