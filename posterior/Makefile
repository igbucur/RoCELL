MINGW_DIR = /c/rtools40/mingw64
MULTINEST_DIR = ../MultiNest/MultiNest_v3.12

FC = gfortran
CC = gcc
CXX = g++
FFLAGS += -ffree-line-length-none -O3 -I../src
CFLAGS += -O3 -I$(MINGW_DIR)/include
LDFLAGS = -L$(MINGW_DIR)/lib 
LAPACKLIB = -llapack -lblas


# CPPLIBS = $(shell mpic++ --showme:link)
LIBS = -L$(MULTINEST_DIR) -lnest3 -llapack -lstdc++ -lblas # $(shell mpif90 --showme:link) 
# LIBS =  -L$(MULTINEST_DIR) -lnest3 $(LAPACKLIB) -lstdc++
 
OBJFILES = RoCELL.o RoCELL_MultiNest.o


default: RoCELL

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $*.cpp

RoCELL.so: 
	g++ -o libRoCELL.so RoCELL.o -shared

RoCELL: $(OBJFILES)
	$(FC) $(FFLAGS) -o ../RoCELL $(OBJFILES) $(LIBS)
#	$(FC) $(FFLAGS) -nofor-main -o ../eggboxC++ $(OBJFILES) $(LIBS)

clean:
	rm -f *.o *.so *.mod ../RoCELL
