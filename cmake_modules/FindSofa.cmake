# Find SOFA
# ~~~~~~~~
# Copyright (c) 2014, Gurprit Singh
# Redistribution and use is allowed.
#
# Modified by Pedro Fluxa (PUC,2019)
# Modification 1: the script only looks for SOFA_cxx.
#
# SOFA package includes following libraries:
# SOFA_cxx cxxsupport sharp fftpack c_utils  /*Order of these libraries is very important to resolve dependencies*/
#
# Once run this will define:
#
# SOFA_FOUND:	        system has SOFA lib
# SOFA_LIBRARIES:	    full path to the SOFA package libraries
# SOFA_INCLUDE_DIR:	    where to find headers
#

FIND_PATH(SOFA_INCLUDE_DIR
NAMES
        sofa.h 
        sofam.h
PATHS
        ~/.local/include
        /usr/include/
        /usr/local/include/
)

FIND_LIBRARY(SOFA_LIBRARIES
NAMES
        sofa_c
PATHS
        ~/.local/lib
        /usr/lib/x86_64-linux-gnu/
        /usr/local/lib/
        /usr/local/lib/x86_64-linux-gnu/
)


SET(SOFA_FOUND FALSE)
IF(SOFA_INCLUDE_DIR AND SOFA_LIBRARIES)
	SET(SOFA_LIBRARIES ${SOFA_LIBRARIES})
	SET(SOFA_FOUND TRUE)
	
	MESSAGE(STATUS "SOFA Found!")
	MESSAGE(STATUS "SOFA_INCLUDE_DIR=${SOFA_INCLUDE_DIR}")
	MESSAGE(STATUS "SOFA_LIBRARIES=${SOFA_LIBRARIES}")
ENDIF()


MARK_AS_ADVANCED(
    SOFA_INCLUDE_DIR
    SOFA_LIBRARIES
    SOFA_FOUND
)
