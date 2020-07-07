# RoCELL
Implementation of the approach described in "Robust Causal Estimation in the Large-Sample Limit without Strict Faithfulness"

## Prerequisites

Just for installing the software and running basic examples, you will need:

- [GNU Make Build Tool](https://www.gnu.org/software/make/)
- [Armadillo C++ Library for Linear Algebra](http://arma.sourceforge.net/)
- [LAPACK Linear Algebra Package](http://www.netlib.org/lapack/)
- [OpenBLAS Linear Algebra Kernels](https://www.openblas.net/)

This software is routinely available on Linux-based software. For example, the software can be installed on Ubuntu using the following terminal command:
`sudo apt install make libarmadillo-dev liblapack-dev libopenblas-dev`.

This software can also run on Windows, but requires quite a bit more work to install all the prerequisites. One workable option would be to install the [MinGW](http://www.mingw.org/) environment. 


## Installation Instructions

Download the software from GitHub with the following command:
`git clone --recurse-submodules https://github.com/igbucur/RoCELL.git`.

Build the MultiNest nested sampling program and RoCELL by simply running `make` in the root directory. It is likely that some compiler flags will have to be set individually, so that the C++ compiler, e.g., [gcc](https://gcc.gnu.org/), can find the required headers and libraries.

To verify that the software has been built successfully, one can use the provided script by running `bash ./run_IV_example.sh` in a Linux or Windows PowerShell terminal. 

