## Description
In the era of big data, the increasing availability of huge data sets can
paradoxically be harmful when our causal inference method is designed to search 
for a causal model that is faithful to our data. Under the commonly made Causal 
Faithfulness Assumption, we look for patterns of dependencies and independencies 
in the data and match them with causal models that imply the same patterns.
However, given enough data, we start picking up on the fact that everything is 
ultimately connected. These interactions are not normally picked up in small samples.
The only faithful causal model in the limit of a large number of samples 
(the large-sample limit) therefore becomes the one where everything is connected. 
Alas, we cannot extract any useful causal information from a completely connected 
structure without making additional (strong) assumptions. We propose an alternative 
approach (RoCELL) that replaces the Causal Faithfulness Assumption with a prior 
that reflects the existence of many "weak" (irrelevant) and "strong" interactions.
RoCELL outputs a posterior distribution over the target causal effect estimator
that leads to good estimates even in the large-sample limit.

## Content

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
which is owned and copyrighted by Farhan Feroz and Mike Hobson. The MultiNest
source code is bundled (as-is, except for build and install configuration) as a
submodule in this package. For more details, please see the README accompanying 
the MultiNest submodule.

## Structure

The code is structured on the skeleton of an [R package](https://r-pkgs.org/index.html) 
package as follows:

- The folder `data` contains pre-saved simulated data, which we use to recreate
the figures from the article. The simulated data can also be reproduced using 
the `aistats-article-figures.R` script in the main folder. The simulated data
sets are described in `R/data.R`.

- The folder `MultiNest` contains the [MultiNest](https://github.com/farhanferoz/MultiNest)
*submodule*, which is a fork of the project developed by Farhan Feroz and Mike Hobson.
The MultiNest submodule implements a complex nested sampling algorithm, which we
employ in RoCELL for estimating model evidences and producing posterior samples.
The original implementation is kept unchanged, with the exception of changes
in the build configuration file made for easier compilation and installation.

- The folder `R` contains the R files necessary for reproducing the figures from
the article.

- The folder `man` contains the documentation for the implemented functions.

- The folder `posterior` contains the C++ implementation of RoCELL written for
integration with MultiNest.

- The folder `src` contains an Rcpp wrapper to the RoCELL implementation in
`posterior`. The wrapper functions are called by `aistats-article-figures.R`.

- The folder `scripts` contains the script `aistats-article-figures.R`, which
can be run from R to produce the figures in the AISTATS 2017 article and a
basic script (`run_IV_example.sh`) for verifying that MultiNest was installed
correctly.

- The top folder also contains the following files:
  - `DESCRIPTION` is the file describing the R package
  - `NAMESPACE` is the file specifying the fucntions provided by the R package
  - `LICENSE.md` is the file containing the GPL-3 license


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
script by running `bash scripts/run_IV_example.sh` in a Linux or Windows terminal
(requires [GNU Bash](https://www.gnu.org/software/bash/)). To verify that the
R package is installed successfully, run `Rscript scripts/aistats-article-figures.R`.

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