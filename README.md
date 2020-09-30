# Description
The data set contains source code implementing the RoCELL algorithm, which is 
described in the article titled "[Robust Causal Estimation in the Large-Sample
Limit without Strict Faithfulness](http://proceedings.mlr.press/v54/bucur17a.html)" 
by Ioan Gabriel Bucur, Tom Claassen and Tom Heskes. This research is presented 
in Chapter 2 of the PhD thesis titled "Being Bayesian about Causal Inference" by
Ioan Gabriel Bucur.

The code is written in the R and C++ programming languages. RoCELL makes use of
the [MultiNest](https://github.com/farhanferoz/MultiNest) nested sampling algorithm, 
which is owned and copyrighted by Farhan Feroz and Mike Hobson. For more details 
please see the LICENCE accompanying the MultiNest submodule.

The code is structured on the skeleton of an [R package](https://r-pkgs.org/index.html) 
package as follows.

- The folder `data` contains pre-saved simulated data, which we use to.
The simulated data can be reproduced using the 

- The folder `man` contains the documentation documentation f
Implementation of the approach described in 



## Prerequisites

Just for installing the software and running basic examples, you will need:

- [GNU Bash](https://www.gnu.org/software/bash/)
- [GNU Make Build Tool](https://www.gnu.org/software/make/)
- [GNU C++ Compiler](https://gcc.gnu.org/) or similar
- [GNU Fortran Compiler](https://gcc.gnu.org/fortran/) or similar
- [Armadillo C++ Library for Linear Algebra](http://arma.sourceforge.net/)
- [LAPACK Linear Algebra Package](http://www.netlib.org/lapack/)
- [OpenBLAS Linear Algebra Kernels](https://www.openblas.net/)
- [Boost C++ Libraries](https://www.boost.org/)

These prerequisites are routinely available for Linux-based environments. For 
example, the (missing) prerequisites can be installed on Ubuntu using the following 
terminal command: `sudo apt install make lib{armadillo|boost|lapack|openblas}-dev`.

This software can also run on Windows, but typically requires quite a bit more 
work for installing all the prerequisites. The simplest way to make it work 
would be to install the most recent version of the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) (WSL). The
WSL can be used to emulate a Linux distribution like Ubuntu on a machine running 
Windows, so the prerequisites can be installed as if in a Linux-based environment.

A slightly more complicated but workable and native solution would be to employ 
the [MSYS2](https://www.msys2.org/) build platform. This setup involves installing
packages (including necessary dependencies) from the MSYS2 collection: `pacman -S mingw-w64-x86_64-{armadillo|boost|gcc-fortran|hdf5|lapack|openblas}`.


## Installation Instructions

Download the software from GitHub with the following command:
`git clone --recurse-submodules https://github.com/igbucur/RoCELL.git`.

Build the MultiNest nested sampling program and RoCELL by simply running `make` 
in the root directory. It is likely that some compiler flags will have to be set 
individually, so that the C++ compiler, e.g., [gcc](https://gcc.gnu.org/), can 
find the required headers and libraries. On Windows, the path to the collection
of prerequisites must be specified separately. For example, if using 
[MSYS2](https://www.msys2.org/) with 64-bit [MinGW](http://www.mingw.org/), then 
one must add `-I/c/msys64/mingw64/include` to `CFLAGS` and `-L/c/msys64/mingw64/lib` 
to `LDFLAGS` in the Makefile, assuming default installation directories.

To verify that the software has been built successfully, one can use the provided 
script by running `bash ./run_IV_example.sh` in a Linux or Windows terminal
(requires [GNU Bash](https://www.gnu.org/software/bash/)).

