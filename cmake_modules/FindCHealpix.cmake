# Find HealPix
# ~~~~~~~~
# Copyright (c) 2014, Gurprit Singh
# Redistribution and use is allowed.
#
# Modified by Pedro Fluxa (PUC,2019)
# Modification 1: the script only looks for healpix_cxx.
#
# CHEALPIX package includes following libraries:
# healpix_cxx cxxsupport sharp fftpack c_utils  /*Order of these libraries is very important to resolve dependencies*/
#
# Once run this will define:
#
# CHEALPIX_FOUND:	        system has CHEALPIX lib
# CHEALPIX_LIBRARIES:	    full path to the CHEALPIX package libraries
# CHEALPIX_INCLUDE_DIR:	    where to find headers
#

FIND_PATH(CHEALPIX_INCLUDE_DIR
NAMES
        chealpix.h
PATHS
        ~/.local/include/
)

FIND_LIBRARY(CHEALPIX_LIBRARIES
NAMES
        chealpix
PATHS
        ~/.local/lib
)


SET(CHEALPIX_FOUND FALSE)
IF(CHEALPIX_INCLUDE_DIR AND CHEALPIX_LIBRARIES)
	SET( CHEALPIX_LIBRARIES ${CHEALPIX_LIBRARIES}
		${CHEALPIX_CXXSUPPORT_LIBRARIES}
		${CHEALPIX_SHARP_LIBRARIES}
		${CHEALPIX_FFTPACK_LIBRARIES}
		${CHEALPIX_CUTILS_LIBRARIES} )
		
	SET(CHEALPIX_FOUND TRUE)
	
	MESSAGE(STATUS "HealPix Found!")
	MESSAGE(STATUS "CHEALPIX_INCLUDE_DIR=${CHEALPIX_INCLUDE_DIR}")
	MESSAGE(STATUS "CHEALPIX_LIBRARIES=${CHEALPIX_LIBRARIES}")
ENDIF()


MARK_AS_ADVANCED(
CHEALPIX_INCLUDE_DIR
CHEALPIX_LIBRARIES
CHEALPIX_FOUND
)
