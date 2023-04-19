#!/bin/bash
source /opt/intel/oneapi/setvars.sh intel64
cd sofa/20210512/c/src
cp makefile makefile.backup
sed s%"INSTALL_DIR = \$(HOME)"%"INSTALL_DIR = /usr"%g makefile > _makefile1
sed s%"CFLAGF = -c -pedantic -Wall -O"%"CFLAGF = -c -pedantic -Wall -O -fPIC"%g _makefile1 > _makefile2
sed s%"CFLAGX = -pedantic -Wall -O"%"CFLAGX = -pedantic -Wall -O -fPIC"%g _makefile2 > _makefile3
sed s%"SOFA_LIB_NAME = libsofa_c.a"%"SOFA_LIB_NAME = libsofa_c.so"%g _makefile3 > _makefile4
sed s%"CCOMPC = gcc"%"CCOMPC = icx"%g _makefile4 > _makefile5
cp _makefile5 makefile
make -j8
make install
make test
rm _makefile*
