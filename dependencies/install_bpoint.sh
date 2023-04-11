#!/bin/bash
source /opt/intel/oneapi/setvars.sh intel64
cd bpoint
mkdir build
cd build/
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx ../
make
make install

# create binary lookup table with ephemeris from 2000 to 2040
cd /pisco4cmb/deps/bpoint/data
cat header.405 > ephem.txt
cat ascp2000.405 >> ephem.txt
cat ascp2020.405 >> ephem.txt
cat ascp2040.405 >> ephem.txt
/usr/bin/asc2bin.x ephem.txt
rm ephem.txt
cp DEc405 /pisco4cmb/data/
