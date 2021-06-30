cmake_minimum_required(VERSION 3.12)
# Tell Cmake about where to find scripts to look up for some dependencies
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake_modules/")
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
project(pisco
        LANGUAGES CUDA CXX
        VERSION 0.2
        DESCRIPTION "Compiles PIxel Space COnvolution library.")
# enable compilation using standard C++11
set(CMAKE_CXX_STANDARD 11)
# debug flags
set(CMAKE_CFLAGS_DEBUG "-g -G -Wall -O0")
set(CMAKE_CXXFLAGS_DEBUG "-g -G -Wall -O0")
# directives to find required packages
find_package(OpenMP REQUIRED)
find_package(Healpix REQUIRED)
find_package(LAPACK REQUIRED)
# some day, I will add a proper Input/Output module using this...
#find_package(cnpy REQUIRED)
find_package(Sofa REQUIRED)
find_package(cfitsio REQUIRED)
# it seems like CMAKE can't find openmp on its own
# compile with -O2 seems to be a good compromise
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -pthread -O3")
# flag to disable correct polarization angle calculation
if(CONVOLVER_DISABLECHI)
  add_definitions(-DCONVOLVER_DISABLECHI)
endif()
# flag to enable dumping beams to a file
if(POLBEAM_DUMPBEAMS)
  add_definitions(-DPOLBEAM_DUMPBEAMS)
endif()
# flag to enable parallelization
if(CONVOLVER_OPENMPENABLED)
  add_definitions(-DCONVOLVER_OPENMPENABLED)
endif()
# flag to enable debug print statements in mapping
if(MAPPING_DEBUG)
  add_definitions(-DMAPPING_DEBUG)
endif()
# flag to enable debug print statements in convovler
if(CONVOLVER_DEBUG)
  add_definitions(-DCONVOLVER_DEBUG)
endif()
# flag to enable dumping beams to text files
if(POLBEAM_DUMPBEAMS)
  add_definitions(-DPOLBEAM_DUMPBEAMS)
endif()
# enable gpu to compute the beam/sky convolution
if(USE_CUDA)
  find_package(CUDA REQUIRED)
  set(CUDA_SEPARABLE_COMPILATION ON)
  # instruct nvcc to use fPIC for separate compilation of CUDA objects
  set(CUDA_NVCC_FLAGS 
    ${CUDA_NVCC_FLAGS}; 
    --default-stream per-thread
    -arch=sm_52
    --compiler-options -fPIC)
endif()
# enable beam interpolation when calculating the convolution
# roughly doubles the time taken by the gpu routine!
if(CUDACONV_USE_INTERPOLATION)
  add_definitions(-DCUDACONV_USE_INTERPOLATION)
endif()
# use GPU texture memory to store the sky map
# speed up things in some cases.
if(CUDACONV_SKYASTEXTURE)
  add_definitions(-DCUDACONV_SKYASTEXTURE)
endif()

# compile main library (libpisco)
if(USE_CUDA)
    cuda_add_library(pisco SHARED
        src/Sky/sky.cpp
        src/SkyMap/skymap.cpp
        src/Scan/scan.cpp
        src/Bpoint/bpoint.cpp
        src/Bpoint/ElapsedTime.cpp
        src/Bpoint/jpl_eph.cpp
        src/Polbeam/polbeam.cpp
        src/Sphtrigo/sphtrigo.cpp
        src/GpuConvolver/gpuconvolver.cu
        src/Convolver/convolver.cpp
        src/GpuConvolver/cuda/cudaConv2.cu
        src/GpuConvolver/cuda/cudaSphtransf.cu
        src/GpuConvolver/cuda/cudaHealpix.cu)
else()
    add_library(pisco SHARED
        src/Sky/sky.cpp
        src/SkyMap/skymap.cpp
        src/Convolver/convolver.cpp
        src/Scan/scan.cpp
        src/Bpoint/bpoint.cpp
        src/Bpoint/ElapsedTime.cpp
        src/Bpoint/jpl_eph.cpp
        src/Polbeam/polbeam.cpp
        src/Sphtrigo/sphtrigo.cpp)
endif()
# include private headers
target_include_directories(pisco PRIVATE 
    src/
    ${HEALPIX_INCLUDE_DIR})
# include public headers from dependencies
target_include_directories(pisco PUBLIC
    ${SOFA_INCLUDE_DIR}
    ${OpenMP_INCLUDE_DIRS}
    ${HEALPIX_INCLUDE_DIR}
    ${CHEALPIX_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR})
# compile wholesky simulation executable
if(USE_CUDA)
  cuda_add_executable(simulate_wholesky_scan.x programs/cuda_sim_wholesky_scan.cpp)
  cuda_add_executable(simulate_pointsource_obs.x programs/cuda_sim_pointsource_obs.cpp)
else()
  add_executable(simulate_wholesky_scan.x programs/sim_wholesky_scan.cpp)
endif()
target_include_directories(simulate_wholesky_scan.x PRIVATE src/)
target_link_libraries(simulate_wholesky_scan.x 
    pisco
    ${SOFA_LIBRARIES}
    ${OpenMP_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${CHEALPIX_LIBRARIES}
    ${HEALPIX_LIBRARIES}
    ${CFITSIO_LIBRARIES})
    
target_include_directories(simulate_pointsource_obs.x PRIVATE src/)
target_link_libraries(simulate_pointsource_obs.x 
    pisco
    ${SOFA_LIBRARIES}
    ${OpenMP_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${CHEALPIX_LIBRARIES}
    ${HEALPIX_LIBRARIES}
    ${CFITSIO_LIBRARIES})

