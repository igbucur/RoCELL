FC = gfortran
CC = gcc
CXX = g++
FFLAGS += -O3 -ffree-line-length-none
CFLAGS += -O3 -I/c/rtools40/mingw64/include -I$(R_LIBS)/RcppArmadillo/include
LDFLAGS = -L/c/rtools40/mingw64/lib 
LAPACKLIB = -llapack -lblas

R_LIBS = $(HOME)/Documents/R/win-library/4.0
MULTINEST_DIR = MultiNest/Multinest_v3.12

export FC CC CXX FFLAGS CFLAGS LDFLAGS LAPACKLIB

default: build_MultiNest RoCELL
clean: clean_RoCELL clean_MultiNest 

build_MultiNest: 
	make -C $(MULTINEST_DIR) default

clean_MultiNest: 
	make -C $(MULTINEST_DIR) clean

RoCELL:
	make -C posterior
	
clean_RoCELL:
	make -C posterior clean
	

