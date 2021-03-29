cmake_minimum_required(VERSION 3.9)

project(pixconv
        LANGUAGES CXX
        VERSION 0.1
        DESCRIPTION "Pixel space convolution on the sphere.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")

# enable compilation using standard C++11
set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)
find_package(Healpix REQUIRED)
find_package(LAPACK REQUIRED)
find_package(cnpy REQUIRED)
find_package(Sofa REQUIRED)
find_package(cfitsio REQUIRED)

# it seems like CMAKE can't find openmp on its own...
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
# debug with -O0
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

# options to compile the cuda code separately
#set(CUDA_SEPARABLE_COMPILATION OFF)
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#set(CUDA_HOST_COMPILER g++)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread" )

# executable
add_executable(test.exe
    src/main.cpp
)

# shared library
add_library(libpisco SHARED
    src/Sky/sky.cpp
    src/Scan/scan.cpp
    src/Bpoint/bpoint.cpp
    src/Bpoint/ElapsedTime.cpp
    src/Bpoint/jpl_eph.cpp
    src/Mapping/healpix_map.cpp
    src/Mapping/mapping_routines.cpp
    src/Mapping/chealpix_interpolate.cpp
    src/Polbeam/polbeam.cpp
    src/Sphtrigo/sphtrigo.cpp
    src/Convolver/convolver.cpp
)

target_include_directories(libpisco PRIVATE
    ${CNPY_INCLUDE_DIR}
    ${SOFA_INCLUDE_DIR}
    ${OpenMP_INCLUDE_DIRS}
    ${HEALPIX_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR}
)

target_include_directories(libpisco PUBLIC
	src/
)

target_include_directories(test.exe PRIVATE
    ${CNPY_INCLUDE_DIR}
    ${SOFA_INCLUDE_DIR}
    ${OpenMP_INCLUDE_DIRS}
    ${HEALPIX_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR}
)

target_include_directories(test.exe PUBLIC
	src/
)

target_link_libraries(test.exe
    libpisco
    ${CNPY_LIBRARIES}
    ${SOFA_LIBRARIES}
    ${OpenMP_LIBRARIES}
    ${HEALPIX_LIBRARIES}
    ${CFITSIO_LIBRARIES}
    -lchealpix
    -lopenblas
)

