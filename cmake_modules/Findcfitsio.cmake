# Find CFITSIO
# ~~~~~~~~
# Copyright (c) 2014, Gurprit Singh
# Redistribution and use is allowed.
#
# Modified by Pedro Fluxa (PUC,2019)
# Modification 1: the script only looks for CFITSIO_cxx.
#
# CFITSIO package includes following libraries:
# CFITSIO_cxx cxxsupport sharp fftpack c_utils  /*Order of these libraries is very important to resolve dependencies*/
#
# Once run this will define:
#
# CFITSIO_FOUND:	        system has CFITSIO lib
# CFITSIO_LIBRARIES:	    full path to the CFITSIO package libraries
# CFITSIO_INCLUDE_DIR:	    where to find headers
#

FIND_PATH(CFITSIO_INCLUDE_DIR
NAMES
        fitsio2.h 
        fitsio.h
PATHS
        ~/.local/include
        /usr/include/
        /usr/local/include/
)

FIND_LIBRARY(CFITSIO_LIBRARIES
NAMES
        cfitsio
PATHS
        ~/.local/lib
        /usr/lib/x86_64-linux-gnu/
        /usr/local/lib/
        /usr/local/lib/x86_64-linux-gnu/
)


SET(CFITSIO_FOUND FALSE)
IF(CFITSIO_INCLUDE_DIR AND CFITSIO_LIBRARIES)
	SET(CFITSIO_LIBRARIES ${CFITSIO_LIBRARIES})
	SET(CFITSIO_FOUND TRUE)
	
	MESSAGE(STATUS "CFITSIO Found!")
	MESSAGE(STATUS "CFITSIO_INCLUDE_DIR=${CFITSIO_INCLUDE_DIR}")
	MESSAGE(STATUS "CFITSIO_LIBRARIES=${CFITSIO_LIBRARIES}")
ENDIF()


MARK_AS_ADVANCED(
    CFITSIO_INCLUDE_DIR
    CFITSIO_LIBRARIES
    CFITSIO_FOUND
)
