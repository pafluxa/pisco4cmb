# Find CNPY
# ~~~~~~~~
# Copyright (c) 2014, Gurprit Singh
# Redistribution and use is allowed.
#
# Modified by Pedro Fluxa (PUC,2019)
# Modification 1: the script only looks for CNPY_cxx.
#
# CNPY package includes following libraries:
# CNPY_cxx cxxsupport sharp fftpack c_utils  /*Order of these libraries is very important to resolve dependencies*/
#
# Once run this will define:
#
# CNPY_FOUND:	        system has CNPY lib
# CNPY_LIBRARIES:	    full path to the CNPY package libraries
# CNPY_INCLUDE_DIR:	    where to find headers
#

FIND_PATH(CNPY_INCLUDE_DIR
NAMES
        cnpy.h
PATHS
        ~/.local/include
        /usr/include/
        /usr/local/include/
)

FIND_LIBRARY(CNPY_LIBRARIES
NAMES
        cnpy
PATHS
        ~/.local/lib
        /usr/lib/
        /usr/local/lib/
)


SET(CNPY_FOUND FALSE)
IF(CNPY_INCLUDE_DIR AND CNPY_LIBRARIES)
	SET(CNPY_LIBRARIES ${CNPY_LIBRARIES})
	SET(CNPY_FOUND TRUE)
	
	MESSAGE(STATUS "CNPY Found!")
	MESSAGE(STATUS "CNPY_INCLUDE_DIR=${CNPY_INCLUDE_DIR}")
	MESSAGE(STATUS "CNPY_LIBRARIES=${CNPY_LIBRARIES}")
ENDIF()


MARK_AS_ADVANCED(
    CNPY_INCLUDE_DIR
    CNPY_LIBRARIES
    CNPY_FOUND
)
