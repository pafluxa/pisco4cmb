CC = g++
RM = rm
CP = cp
 
# Change these as required
OPTIMIZE = -O3
DEBUG = 
CXXFLAGS = -fopenmp -fPIC -Wall $(OPTIMIZE) $(DEBUG)
CFLAGS = $(CXXFLAGS)
# Flags passed to gcc during linking
LINK = -shared 
 
# Name of the "thing" to build
TARGET = libbpoint.so
 
# Additional linker libraries
LIBS = -lsofa_c -lgomp
 
# Version of the library to build
VERSION = 1
 
# Source code files used in this project
# Add any additional files to this line
 
SRCS = bpoint.cpp jpl_eph.cpp
 
# -------------------------------------------------------------
# Nothing should need changing below this line
 
OBJS = $(SRCS:.cpp=.o)
# Rules for building
$(TARGET): $(OBJS)
	$(CC) $(LINK) -o $@ $^ $(LIBS)
 
init.o: $(TARGET)_rev.h
 
.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS)
 
.PHONY: install
install: $(TARGET)
	$(CP) $(TARGET) LIBS:
 
.PHONY: revision
revision:
	bumprev $(VERSION) $(TARGET)
