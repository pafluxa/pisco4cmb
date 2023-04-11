#!/bin/bash
source /opt/intel/oneapi/setvars.sh intel64
cd Healpix_3.82
export CC=icx
export CXX=icpx
export FC=ifort
./configure --auto=cxx --prefix=/usr/
make -j8
echo "/pisco4cmb/deps/Healpix_3.82/lib" >> /etc/ld.so.conf.d/healpix_cxx.conf
ldconfig
