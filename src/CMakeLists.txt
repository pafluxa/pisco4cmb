cmake_minimum_required(VERSION 3.9)

project( pixconv
		 LANGUAGES CXX CUDA
         VERSION 0.1
         DESCRIPTION "Pixel space convolution on the sphere.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")

# enable compilation using the C++11 standard
set(CMAKE_CXX_STANDARD 11)

find_package( Threads REQUIRED )
find_package( OpenMP REQUIRED )
# find HEALPix. Take a look at the FindHealpix.cmake file for more information.
find_package( Healpix REQUIRED )
find_package( CUDA REQUIRED )

# it seems like CMAKE can't find openmp on its own...
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
# optimize with -O0 for testing
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# options to compile the cuda code separately
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER g++)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread" )

include_directories( 
    ${OpenMP_INCLUDE_DIRS}
    ${HEALPIX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS} 
)

include_directories( gpuconv.exe
	Sky/
	Scan/
    Polbeam/
    Sphtrigo/
    Convolver/
    GpuConvolver/ 
)

CUDA_COMPILE( cudaObj
	"${CMAKE_SOURCE_DIR}/GpuConvolver/cuda/cudaconv.cu" 
)

cuda_add_executable( gpuconv.exe
    gpu_test.cpp
    Sky/sky.cpp
    Scan/scan.cpp
    Polbeam/polbeam.cpp
    Sphtrigo/sphtrigo.cpp
    Convolver/convolver.cpp
    GpuConvolver/gpuconvolver.cpp
    ${cudaObj}
)

# link 
target_link_libraries( gpuconv.exe ${CUDA_LIBRARIES} ${OpenMP_LIBS} ${HEALPIX_LIBRARIES} cfitsio )

# tell cmake where to install stuff
install(TARGETS gpuconv.exe DESTINATION "${CMAKE_SOURCE_DIR}/bin" )
