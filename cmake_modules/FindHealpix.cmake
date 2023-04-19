# Find HealPix
# ~~~~~~~~
# Copyright (c) 2014, Gurprit Singh
# Redistribution and use is allowed.
#
# Modified by Pedro Fluxa (PUC,2019)
# Modification 1: the script only looks for healpix_cxx.
#
# HEALPIX package includes following libraries:
# healpix_cxx cxxsupport sharp fftpack c_utils  /*Order of these libraries is very important to resolve dependencies*/
#
# Once run this will define:
#
# HEALPIX_FOUND:	        system has HEALPIX lib
# HEALPIX_LIBRARIES:	    full path to the HEALPIX package libraries
# HEALPIX_INCLUDE_DIR:	    where to find headers
#

FIND_PATH(HEALPIX_INCLUDE_DIR
NAMES
        healpix_base.h
PATHS
        ~/.local/include/healpix_cxx
		/pisco4cmb/deps/Healpix_3.82/include/healpix_cxx
)

FIND_LIBRARY(HEALPIX_LIBRARIES
NAMES
        healpix_cxx
PATHS
        ~/.local/lib
		/pisco4cmb/deps/Healpix_3.82/lib
)


SET(HEALPIX_FOUND FALSE)
IF(HEALPIX_INCLUDE_DIR AND HEALPIX_LIBRARIES)
	SET( HEALPIX_LIBRARIES ${HEALPIX_LIBRARIES}
		${HEALPIX_CXXSUPPORT_LIBRARIES}
		${HEALPIX_SHARP_LIBRARIES}
		${HEALPIX_FFTPACK_LIBRARIES}
		${HEALPIX_CUTILS_LIBRARIES} )
		
	SET(HEALPIX_FOUND TRUE)
	
	MESSAGE(STATUS "HealPix Found!")
	MESSAGE(STATUS "HEALPIX_INCLUDE_DIR=${HEALPIX_INCLUDE_DIR}")
	MESSAGE(STATUS "HEALPIX_LIBRARIES=${HEALPIX_LIBRARIES}")
ENDIF()


MARK_AS_ADVANCED(
HEALPIX_INCLUDE_DIR
HEALPIX_LIBRARIES
HEALPIX_FOUND
)
