# h/t to @jimhester and @yihui for this parse block:
# https://github.com/yihui/knitr/blob/dc5ead7bcfc0ebd2789fe99c527c7d91afb3de4a/Makefile#L1-L4
# Note the portability change as suggested in the manual:
# https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Writing-portable-packages
PKGNAME = `sed -n "s/Package: *\([^ ]*\)/\1/p" DESCRIPTION`
PKGVERS = `sed -n "s/Version: *\([^ ]*\)/\1/p" DESCRIPTION`

ifeq ($(OS), Windows_NT)
  R_EXEC=/c/Program\ Files/R/R-4.0.2/bin/R.exe
  MINGW_DIR=/c/msys64/mingw64
  CC = $(MINGW_DIR)/bin/gcc.exe
  CXX = $(MINGW_DIR)/bin/g++.exe
  FC = $(MINGW_DIR)/bin/gfortran.exe
  AR = $(MINGW_DIR)/bin/ar.exe r
  LD = $(MINGW_DIR)/bin/ld.exe
else
	R_EXEC = R
	CC = gcc
	CXX = g++
	FC = gfortran
	AR = ar r
	LD = ld
endif

MULTINEST_DIR = MultiNest/Multinest_v3.12

export CC FC AR LD


default: build 

build: build_MultiNest build_RoCELL
	$(R_EXEC) CMD build .
	
clean: clean_RoCELL clean_MultiNest

check: build
	R CMD check --no-manual $(PKGNAME)_$(PKGVERS).tar.gz

install: build
	$(R_EXEC) CMD INSTALL $(PKGNAME)_$(PKGVERS).tar.gz

clean: clean_MultiNest clean_RoCELL
	@rm -rf $(PKGNAME)_$(PKGVERS).tar.gz $(PKGNAME).Rcheck

build_MultiNest: 
	make -C $(MULTINEST_DIR) default

clean_MultiNest: 
	make -C $(MULTINEST_DIR) clean

build_RoCELL:
	make -C posterior
	
clean_RoCELL:
	make -C posterior clean

