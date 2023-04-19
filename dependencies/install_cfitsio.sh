#!/bin/bash
source /opt/intel/oneapi/setvars.sh intel64
export CC=icx
cd cfitsio-4.2.0
./configure --prefix=/usr/
make -j8
make install
