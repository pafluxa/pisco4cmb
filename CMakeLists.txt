cmake_minimum_required(VERSION 3.9)
# Tell Cmake about where to find scripts to look up for some dependencies
set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    "${CMAKE_SOURCE_DIR}/cmake_modules/")
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)

project(pixconv
        LANGUAGES CXX
        VERSION 0.1
        DESCRIPTION "Compiles PIxel Space COnvolution library.")

# enable compilation using standard C++11
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CFLAGS_DEBUG "-g -G -Wall -O0")
set(CMAKE_CXXFLAGS_DEBUG "-g -G -Wall -O0")

find_package(OpenMP REQUIRED)
find_package(Healpix REQUIRED)
find_package(CHealpix REQUIRED)
find_package(LAPACK REQUIRED)
# some day, I will add a proper Input/Output module using this...
#find_package(cnpy REQUIRED)
find_package(Sofa REQUIRED)
find_package(cfitsio REQUIRED)

# it seems like CMAKE can't find openmp on its own...
SET(CMAKE_CXX_FLAGS 
  "${CMAKE_CXX_FLAGS} -march=native -fopenmp -pthread -O2")
 
if(CONVOLVER_DISABLECHI)
  add_definitions(-DCONVOLVER_DISABLECHI)
endif()

if(CONVOLVER_DISABLECHI)
  add_definitions(-DCONVOLVER_DISABLECHI)
endif()

 
# library 
add_library(pisco SHARED
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
target_include_directories(pisco PUBLIC
	src/
    ${SOFA_INCLUDE_DIR}
    ${OpenMP_INCLUDE_DIRS}
    ${HEALPIX_INCLUDE_DIR}
    ${CHEALPIX_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR}
)
# executable
add_executable(simulate_point_source_scan.x
    src/sim_point_source_scan.cpp
)
target_include_directories(simulate_point_source_scan.x PRIVATE
	src/
)
target_link_libraries(simulate_point_source_scan.x PRIVATE
	pisco
    ${SOFA_LIBRARIES}
    ${OpenMP_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${CHEALPIX_LIBRARIES}
    ${HEALPIX_LIBRARIES}
    ${CFITSIO_LIBRARIES}
)
