# RoCELL
Implementation of the approach described in "Robust Causal Estimation in the Large-Sample Limit without Strict Faithfulness"

## Prerequisites

Just for installing the software and running basic examples, you will need:

- [GNU Make Build Tool](https://www.gnu.org/software/make/)
- [GNU C++ Compiler](https://gcc.gnu.org/) or similar
- [GNU Fortran Compiler](https://gcc.gnu.org/fortran/) or similar
- [Armadillo C++ Library for Linear Algebra](http://arma.sourceforge.net/)
- [LAPACK Linear Algebra Package](http://www.netlib.org/lapack/)
- [OpenBLAS Linear Algebra Kernels](https://www.openblas.net/)
- [Boost C++ Libraries](https://www.boost.org/)

This software is routinely available on Linux-based environments. For example, the software can be installed on Ubuntu using the following terminal command:
`sudo apt install make libarmadillo-dev libboost-dev liblapack-dev libopenblas-dev`.

This software can also run on Windows, but requires quite a bit more work to install all the prerequisites. One workable option would be to employ the [MSYS2](https://www.msys2.org/) build platform. This setup involves installing packages (including necessary dependencies) from the MSYS2 collection: `pacman -S mingw-w64-x86_64-armadillo mingw-w64-x86_64-boost mingw-w64-x86_64-openblas mingw-w64-x86_64-lapack mingw-w64-x86_64-hdf5 mingw-w64-x86_64-gcc-fortran`.


## Installation Instructions

Download the software from GitHub with the following command:
`git clone --recurse-submodules https://github.com/igbucur/RoCELL.git`.

Build the MultiNest nested sampling program and RoCELL by simply running `make` in the root directory. It is likely that some compiler flags will have to be set individually, so that the C++ compiler, e.g., [gcc](https://gcc.gnu.org/), can find the required headers and libraries. On Windows, the path to the collection of prerequisites must be specified separately. If using [MSYS2](https://www.msys2.org/) with 64-bit [MinGW](http://www.mingw.org/), then one must add `-I/c/msys64/mingw64/include` to `CFLAGS` and `-L/c/msys64/mingw64/lib` to `LDFLAGS` in the Makefile.

To verify that the software has been built successfully, one can use the provided script by running `bash ./run_IV_example.sh` in a Linux or Windows PowerShell terminal. 

