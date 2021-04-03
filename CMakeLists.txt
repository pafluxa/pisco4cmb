cmake_minimum_required(VERSION 3.9)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)

project(pixconv
        LANGUAGES CXX CUDA
        VERSION 0.1
        DESCRIPTION "Pixel space convolution on the sphere.")

# enable compilation using standard C++11
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CUDA_FLAGS "-Wall -lgomp -O3 --default-stream per-thread  -Xcompiler -fopenmp")
#set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -Wall")

find_package(OpenMP REQUIRED)
find_package(Healpix REQUIRED)
find_package(CHealpix REQUIRED)
find_package(LAPACK REQUIRED)
find_package(cnpy REQUIRED)
find_package(Sofa REQUIRED)
find_package(cfitsio REQUIRED)
find_package(CUDA REQUIRED)

# it seems like CMAKE can't find openmp on its own...
SET(CMAKE_CXX_FLAGS 
  "${CMAKE_CXX_FLAGS} -march=native -fopenmp -pthread -O3")
 
# options to compile the cuda code separately
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER g++)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread" )

# executable
cuda_add_executable(wholesky_convolution_gpu.exe
    src/wholesky_convolution_gpu.cpp
)

# executable
cuda_add_executable(wholesky_convolution_cpu.exe
    src/wholesky_convolution_cpu.cpp
)

# shared library
add_library(pisco STATIC
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
    src/GpuConvolver/gpuconvolver.cpp
)

target_include_directories(pisco PRIVATE
    ${CNPY_INCLUDE_DIR}
    ${SOFA_INCLUDE_DIR}
    ${OpenMP_INCLUDE_DIRS}
    ${HEALPIX_INCLUDE_DIR}
    ${CHEALPIX_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR}
)

target_include_directories(pisco PUBLIC
	src/
)

cuda_add_library(cudaconvolution STATIC
    src/GpuConvolver/cuda/beam_times_sky.cu
    src/GpuConvolver/cuda/beam_times_sky.h
    src/GpuConvolver/cuda/healpix_math.cuh
    src/GpuConvolver/cuda/healpix_utils.cuh
    src/GpuConvolver/cuda/sphtrigo.cuh
)

target_include_directories(cudaconvolution PRIVATE
    src/GpuConvolver
)

target_include_directories(
wholesky_convolution_gpu.exe PRIVATE
    ${CNPY_INCLUDE_DIR}
    ${SOFA_INCLUDE_DIR}
    ${OpenMP_INCLUDE_DIRS}
    ${HEALPIX_INCLUDE_DIR}
    ${CHEALPIX_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR}
)

target_include_directories(
wholesky_convolution_cpu.exe PRIVATE
    ${CNPY_INCLUDE_DIR}
    ${SOFA_INCLUDE_DIR}
    ${OpenMP_INCLUDE_DIRS}
    ${HEALPIX_INCLUDE_DIR}
    ${CHEALPIX_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR}
)

target_include_directories(wholesky_convolution_gpu.exe PUBLIC
	src/
)

target_include_directories(wholesky_convolution_cpu.exe PUBLIC
	src/
)

target_link_libraries(wholesky_convolution_cpu.exe
    pisco
    ${CUDA_LIBRARIES}
    ${CNPY_LIBRARIES}
    ${SOFA_LIBRARIES}
    ${OpenMP_LIBRARIES}
    ${HEALPIX_LIBRARIES}
    ${CHEALPIX_LIBRARIES}
    ${CFITSIO_LIBRARIES}
    -lopenblas
)


target_link_libraries(wholesky_convolution_gpu.exe
    pisco
    cudaconvolution
    ${CUDA_LIBRARIES}
    ${CNPY_LIBRARIES}
    ${SOFA_LIBRARIES}
    ${OpenMP_LIBRARIES}
    ${HEALPIX_LIBRARIES}
    ${CHEALPIX_LIBRARIES}
    ${CFITSIO_LIBRARIES}
    -lopenblas
)

