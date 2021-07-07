########################################################################
# CMAKE boiler plate stuff
########################################################################
cmake_minimum_required(VERSION 3.12)
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake_modules/")
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
project(pisco
        LANGUAGES CUDA CXX
        VERSION 0.2
        DESCRIPTION "Compiles PIxel Space COnvolution library.")

########################################################################
# CXX compilation flags
########################################################################
# enable compilation using standard C++11
set(CMAKE_CXX_STANDARD 11)
# debug flags
set(CMAKE_CFLAGS_DEBUG "-g -G -Wall -O0")
set(CMAKE_CXXFLAGS_DEBUG "-g -G -Wall -O0")
# yes, OpenMP is being manually linked
if(CONVOLVER_OPENMPENABLED)
  set(CMAKE_CXX_FLAGS 
    "${CMAKE_CXX_FLAGS} -march=native -fopenmp -pthread -O2")
else()
  set(CMAKE_CXX_FLAGS 
    "${CMAKE_CXX_FLAGS} -march=native -O2")
endif()

########################################################################
# Find dependencies.
########################################################################
# directives to find required packages
find_package(OpenMP REQUIRED)
find_package(Healpix REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Sofa REQUIRED)
find_package(cfitsio REQUIRED)

########################################################################
# Handle compilation flags.
########################################################################
# flag to disable debug messages
if(DEBUG_MESSAGES)
  message(VERBOSE "Compiling with debug messages.")
  add_definitions(-DDEBUG_MESSAGES)
endif()
# flag to enable gpu to compute the beam/sky convolution
if(ENABLE_CUDA)
  message(VERBOSE "Compiling with CUDA support.")
  add_definitions(-DUSE_CUDA)
endif()
# flag to disable correct polarization angle calculation
if(DISABLE_POLANGLE_CORRECTION)
  message(WARNING 
    "POLANGLE_CORRECTION disabled. Convolution results might be incorrect if the beam is too extended.")
  add_definitions(-DDISABLE_POLANGLE_CORRECTION)
endif()
# flag to enable OpenMP
if(USE_OPENMP)
  message(VERBOSE "Compiling with OpenMP support.")
  message(WARNING "Make sure to set OMP_NUM_THREADS flag to avoid using all available CPUs.")
  add_definitions(-DUSE_OPENMP)
endif()

########################################################################
# Compilation
########################################################################
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

