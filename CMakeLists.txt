cmake_minimum_required(VERSION 3.12)
# Tell Cmake about where to find scripts to look up for some dependencies
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_SOURCE_DIR}/cmake_modules/")
# do not build if in source directoru
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# do change/touch source code
set(CMAKE_DISABLE_SOURCE_CHANGES ON)

project(pisco
        LANGUAGES CUDA CXX
        VERSION 0.2
        DESCRIPTION "Compiles PIxel Space COnvolution library.")
# enable compilation using standard C++11
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_C_FLAGS "-O3 -fopenmp")
SET(CMAKE_CXX_FLAGS "-O3 -fopenmp")
# use position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# directives to find required packages
# CUDA is required
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
# OpenMP is required
find_package(OpenMP REQUIRED)
# HealPIX is required (cfitsio too)
find_package(Healpix REQUIRED)
find_package(cfitsio REQUIRED)
# MKL required (20% speed-up!!)
# MKL also provides implementation for dgesv
find_package(MKL REQUIRED)
# SOFA only required if using BPoint
#find_package(Sofa REQUIRED)

# flag to enable debug print statements in mapping
if(MAPPING_DEBUG)
  add_definitions(-DMAPPING_DEBUG)
endif()

# flag to enable debug print statements in convovler
if(CONVOLVER_DEBUG)
  add_definitions(-DCONVOLVER_DEBUG)
endif()

# flag to enable debug print statements in convovler
if(POLBEAM_DEBUG)
  add_definitions(-DPOLBEAM_DEBUG)
endif()

# compile main library (libpisco)
cuda_add_library(pisco SHARED
    src/Sky/sky.cpp
    src/Scanning/scanning.cpp
    src/Mapper/mapper.cpp
    src/Polbeam/polbeam.cpp
    src/Sphtrigo/sphtrigo.cpp
    src/Convolution/convolution_engine.cu
    src/Convolution/cu_utils.cu)

# include private headers
target_include_directories(pisco PRIVATE 
    src/
    ${HEALPIX_INCLUDE_DIR})

# include public headers from dependencies
target_include_directories(pisco PUBLIC
    ${HEALPIX_INCLUDE_DIR}
    ${CHEALPIX_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR})
    
# executable: 
# convolve a point source with a symmetrical gaussian beam of FWHM = 2 deg
cuda_add_executable(point_source_scan.x programs/point_source_scan.cpp)
target_include_directories(point_source_scan.x PRIVATE src/)
target_link_libraries(point_source_scan.x 
    pisco
    ${MKL_LIBRARIES}
    ${HEALPIX_LIBRARIES}
    ${CFITSIO_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY})

# executable: 
# convolve a point source with user provided beams
cuda_add_executable(user_beam_point_source_scan.x programs/user_beam_point_source_scan.cpp)
target_include_directories(user_beam_point_source_scan.x PRIVATE src/)
target_link_libraries(user_beam_point_source_scan.x 
    pisco
    ${MKL_LIBRARIES}
    ${HEALPIX_LIBRARIES}
    ${CFITSIO_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY})

# executable: 
# convolve a user provided sky with gaussian elliptical beams without cross-pol
cuda_add_executable(gaussian_beam_simple_full_sky.x programs/gaussian_beam_simple_full_sky.cpp)
target_include_directories(gaussian_beam_simple_full_sky.x PRIVATE src/)
target_link_libraries(gaussian_beam_simple_full_sky.x 
    pisco
    ${MKL_LIBRARIES}
    ${HEALPIX_LIBRARIES}
    ${CFITSIO_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY})

