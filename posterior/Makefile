FFLAGS += -ffree-line-length-none -O3 -I../src
CFLAGS += -O3 -I$(MINGW_DIR)/include
LDFLAGS += -L$(MINGW_DIR)/lib 
LAPACKLIB = -llapack -lblas

LIBS = -L$(MULTINEST_DIR) -lnest3 -larmadillo -lstdc++ $(LAPACKLIB)
 
OBJFILES = RoCELL_MultiNest.o

default: RoCELL

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $*.cpp

RoCELL: $(OBJFILES)
	$(FC) $(FFLAGS) -o ../RoCELL $(OBJFILES) $(LIBS)

clean:
	rm -f *.o *.mod ../RoCELL
