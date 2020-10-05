# Description
The data set contains source code implementing the RoCELL algorithm, which is 
described in the article titled "[Robust Causal Estimation in the Large-Sample
Limit without Strict Faithfulness](http://proceedings.mlr.press/v54/bucur17a.html)" 
by Ioan Gabriel Bucur, Tom Claassen and Tom Heskes. The data set also contains 
simulated data necessary for exactly reproducing the figures in the article as
well as the routines necessary for recreating it. This research is presented 
in Chapter 2 of the PhD thesis titled "Being Bayesian about Causal Inference" by
Ioan Gabriel Bucur. 

The code is written in the R and C++ programming languages. RoCELL makes use of
the [MultiNest](https://github.com/farhanferoz/MultiNest) nested sampling algorithm, 
which is owned and copyrighted by Farhan Feroz and Mike Hobson. For more details 
please see the LICENCE accompanying the MultiNest submodule.

## Structure

The code is structured on the skeleton of an [R package](https://r-pkgs.org/index.html) 
package as follows:

- The folder `data` contains pre-saved simulated data, which we use to recreate
the figures from the article. The simulated data can also be reproduced using 
the `aistats-article-figures.R` script in the main folder;

- The folder `man` contains the documentation for the implemented functions;

- The folder `MultiNest` contains the MultiNest submodule;

- The folder `R` contains the R files necessary for reproducing the figures from
the article;

- The folder `posterior` contains the C++ implementation of RoCELL written for
integration with MultiNest;

- The folder `src` contains an Rcpp wrapper to the RoCELL implementation in
`posterior`. The wrapper functions are called by `aistats-article-figures.R`.


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

For installing and running the RoCELL R package, a few R packages are also 
required. These are specified in the package `DESCRIPTION` file.


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

To verify that RoCELL has been built successfully, one can use the provided 
script by running `bash ./run_IV_example.sh` in a Linux or Windows terminal
(requires [GNU Bash](https://www.gnu.org/software/bash/)). To verify that the
R package is installed successfully, run `Rscript ./aistats-article-figures.R`.

## Licensing

RoCELL algorithm - Robust Causal Estimation in the Large-Sample Limit without Strict Faithfulness

Copyright (C) 2020 Ioan Gabriel Bucur <ioan.gabriel.bucur@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.