cmake_minimum_required(VERSION 3.15)
project(virta LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OMP "Enable OpenMP" ON)

include_directories(${PROJECT_SOURCE_DIR}/src)

set(EXECUTABLES
    main
)

if(USE_OMP)
    find_package(OpenMP REQUIRED)
endif()

foreach(EXE ${EXECUTABLES})
    add_executable(${EXE}.ex ${EXE}.cpp)

    if(USE_OMP)
        target_link_libraries(${EXE}.ex PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${EXE}.ex PRIVATE USE_OMP)
    endif()

    target_compile_options(${EXE}.ex PRIVATE -O3 -march=native)
endforeach()

find_package(Doxygen REQUIRED)

add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
