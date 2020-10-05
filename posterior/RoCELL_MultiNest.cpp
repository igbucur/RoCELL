#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <algorithm> // for std::find
#include <stdlib.h>
#include "multinest.h"
#include "RoCELL.h"
#include <assert.h>

#include <boost/math/distributions/normal.hpp>
boost::math::normal dist(0.0, 1.0);

// Input configuration structure
typedef struct {
  arma::mat covariance; // data covariance matrix (3x3)
  double slab_precision; // precision of slab component
  double spike_precision; // precision of spike component
  // int model; // 0 - Slab, 1 - Spike, 2 - Reverse
  double w32; // mixture weight for sb32 coefficient
} Config;

/******************************************** loglikelihood routine ****************************************************/

// MultiNest wrapper for RoCELL log-likelihood
//
// Input arguments
// ndim 						= dimensionality (total number of free parameters) of the problem
// npars 						= total number of free plus derived parameters
// context						void pointer, any additional information
//
// Input/Output arguments
// Cube[npars] 			= on entry has the ndim parameters in unit-hypercube
//	 						on exit, the physical parameters plus copy any derived parameters you want to store with the free parameters
//	 
// Output arguments
// lnew 						= loglikelihood


//' MultiNest Log-Likelihood function (w.r.t. sc24, sc34)
void LogLike(double *Cube, int &ndim, int &npars, double &lnew, void *context)
{
  // Read Config structure
  Config* config = (Config*) context;
   
  // Read scaled confounding coefficients, which are N(0, 1) distributed.
  double sc24 = quantile(dist, Cube[0]);
  double sc34 = quantile(dist, Cube[1]);

  // derive structural parameters and compute log-likelihood hessian
  arma::vec spar = get_params_scale_free(config->covariance, sc24, sc34);

  lnew = log_posterior_RoCELL(config->covariance, 1, spar[0], spar[1], spar[2], sc24, sc34, spar[3], spar[4], spar[5],
                              config->slab_precision, config->spike_precision, config->w32);
}

/***********************************************************************************************************************/




/************************************************* dumper routine ******************************************************/

// The dumper routine will be called every updInt*10 iterations
// MultiNest doesn not need to the user to do anything. User can use the arguments in whichever way he/she wants
//
//
// Arguments:
//
// nSamples 						= total number of samples in posterior distribution
// nlive 						= total number of live points
// nPar 						= total number of parameters (free + derived)
// physLive[1][nlive * (nPar + 1)] 			= 2D array containing the last set of live points (physical parameters plus derived parameters) along with their loglikelihood values
// posterior[1][nSamples * (nPar + 2)] 			= posterior distribution containing nSamples points. Each sample has nPar parameters (physical + derived) along with the their loglike value & posterior probability
// paramConstr[1][4*nPar]:
// paramConstr[0][0] to paramConstr[0][nPar - 1] 	= mean values of the parameters
// paramConstr[0][nPar] to paramConstr[0][2*nPar - 1] 	= standard deviation of the parameters
// paramConstr[0][nPar*2] to paramConstr[0][3*nPar - 1] = best-fit (maxlike) parameters
// paramConstr[0][nPar*4] to paramConstr[0][4*nPar - 1] = MAP (maximum-a-posteriori) parameters
// maxLogLike						= maximum loglikelihood value
// logZ							= log evidence value from the default (non-INS) mode
// INSlogZ						= log evidence value from the INS mode
// logZerr						= error on log evidence value
// context						void pointer, any additional information

void dumper(int &nSamples, int &nlive, int &nPar, double **physLive, double **posterior, double **paramConstr, double &maxLogLike, double &logZ, double &INSlogZ, double &logZerr, void *context)
{
	// convert the 2D Fortran arrays to C++ arrays
	
	
	// the posterior distribution
	// postdist will have nPar parameters in the first nPar columns & loglike value & the posterior probability in the last two columns
	
	int i, j;
	
	double postdist[nSamples][nPar + 2];
	for( i = 0; i < nPar + 2; i++ )
		for( j = 0; j < nSamples; j++ )
			postdist[j][i] = posterior[0][i * nSamples + j];
	
	
	
	// last set of live points
	// pLivePts will have nPar parameters in the first nPar columns & loglike value in the last column
	
	double pLivePts[nlive][nPar + 1];
	for( i = 0; i < nPar + 1; i++ )
		for( j = 0; j < nlive; j++ )
			pLivePts[j][i] = physLive[0][i * nlive + j];
}

/***********************************************************************************************************************/




/************************************************** Main program *******************************************************/


char* get_cmd_option(char **begin, char **end, const std::string& option) {
  
  char **itr = std::find(begin, end, option);
  
  if (itr != end && ++itr != end) return *itr;
  
  return 0;
}

bool exists_cmd_option(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}



int main(int argc, char *argv[])
{

	// set the MultiNest sampling parameters
	int IS = 0;					// do Nested Importance Sampling?
	
	int mmodal = 1;					// do mode separation?
	
	int ceff = 0;					// run in constant efficiency mode?
	
	int nlive = 1000;				// number of live points
	
	double efr = 0.8;				// set the required efficiency
	
	double tol = 0.5;				// tol, defines the stopping criteria
	
	int ndims = 2;					// dimensionality (no. of free parameters)
	
	int nPar = 2;					// total no. of parameters including free & derived parameters
	
	int nClsPar = 2;				// no. of parameters to do mode separation on
	
	int updInt = 1000;				// after how many iterations feedback is required & the output files should be updated
							// note: posterior files are updated & dumper routine is called after every updInt*10 iterations
	
	double Ztol = -1E90;				// all the modes with logZ < Ztol are ignored
	
	int maxModes = 100;				// expected max no. of modes (used only for memory allocation)
	
	int pWrap[ndims];				// which parameters to have periodic boundary conditions?
	for(int i = 0; i < ndims; i++) pWrap[i] = 0;
	
  char root[100] = "output_RoCELL-";			// root for output files

	int seed = -1;					// random no. generator seed, if < 0 then take the seed from system clock
	
	int fb = 1;					// need feedback on standard output?
	
	int resume = 0;					// resume from a previous job?
	
	int outfile = 1;				// write output files?
	
	int initMPI = 1;				// initialize MPI routines?, relevant only if compiling with MPI
							// set it to F if you want your main program to handle MPI initialization
	
	double logZero = -1E90;				// points with loglike < logZero will be ignored by MultiNest
	
	int maxiter = 0;				// max no. of iterations, a non-positive value means infinity. MultiNest will terminate if either it 
							// has done max no. of iterations or convergence criterion (defined through tol) has been satisfied
	
	// Read configuration from command line arguments
  Config config;
  
  
  if(exists_cmd_option(argv, argv + argc, "-h")) {
    
    const char* text = "\n"
    "Usage: \n"
    "\tRoCELL [options] -s11 . -s12 . -s13 . -s22 . -s23 . -s33 . \n"
    "\n"
    "Example:\n"
    "./RoCELL -n 500 -s11 1 -s12 1 -s13 1 -s22 2 -s23 2 -s33 3"
    "\n"
    "Options [default value]: \n"
    "\t-s11 (required)  Variance of X_1.\n"
    "\t-s12 (required)  Covariance of X_1 and X_2.\n"
    "\t-s13 (required)  Covariance of X_1 and X_3.\n"
    "\t-s22 (required)  Variance of X_2.\n"
    "\t-s23 (required)  Covariance of X_2 and X_3.\n"
    "\t-s33 (required)  Variance of X_3.\n"
    "\t-n               Number of live points. [1000] \n"
    "\t-o               Root of RoCELL output files. [output_RoCELL] \n"
    "\t-e               MultiNest sampling efficiency. [0.8] \n"
    "\t-w               Spike-and-slab mixture weight for causal effect (sb32). [0.5] \n"
    "\t-slab            Precision of spike component. [1] \n"
    "\t-spike           Precision of spike component. [100] \n"
    "\t-h               Show this screen. \n";
    
    
    std::cout << text << std::endl;                                                                                     
  
    
    std::cout << "Number of live points" << nlive << std::endl;
    
    std::exit(0);
  }
  
  if (exists_cmd_option(argv, argv + argc, "-o")) {
    sprintf(root, "%s-", get_cmd_option(argv, argv + argc, "-o"));
  }
  
  if (exists_cmd_option(argv, argv + argc, "-e")) {
    efr = strtod(get_cmd_option(argv, argv + argc, "-e"), NULL);
    assert(("Sampling efficiency is between 0 and 1", (0 <= efr && efr <= 1)));
  }
  
  if (exists_cmd_option(argv, argv + argc, "-n")) {
    nlive = atoi(get_cmd_option(argv, argv + argc, "-n"));
    assert(("Number of live points is positive", nlive > 0));
  }
  
  if (exists_cmd_option(argv, argv + argc, "-w")) {
    config.w32 = strtod(get_cmd_option(argv, argv + argc, "-w"), NULL);
    assert(("Mixture weight must be between 0 and 1", (0 <= config.w32 && config.w32 <= 1)));
  } else {
    config.w32 = 0.5;
  }
  
  if (exists_cmd_option(argv, argv + argc, "-slab")) {
    config.slab_precision = strtod(get_cmd_option(argv, argv + argc, "-slab"), NULL);
    assert(("Slab precision is positive", config.slab_precision > 0));
  } else {
    config.slab_precision = 1;
  }
  
  if (exists_cmd_option(argv, argv + argc, "-spike")) {
    config.spike_precision = strtod(get_cmd_option(argv, argv + argc, "-spike"), NULL);
    assert(("Spike precision is positive", config.spike_precision > 0));
  } else {
    config.spike_precision = 100;
  }
  
  config.covariance = arma::mat(3, 3, arma::fill::zeros);
  
  if (exists_cmd_option(argv, argv + argc, "-s11")) {
    config.covariance(0, 0) = strtod(get_cmd_option(argv, argv + argc, "-s11"), NULL);
    assert(("Variance of X_1 is positive", config.covariance(0, 0) > 0));
  } else {
    std::cout << "Argument for -s11 not specified. See usage with option -h." << std::endl;
    std::exit(1);
  }
  
  if (exists_cmd_option(argv, argv + argc, "-s12")) {
    config.covariance(0, 1) = config.covariance(1, 0) = strtod(get_cmd_option(argv, argv + argc, "-s12"), NULL);
  } else {
    std::cout << "Argument for -s12 not specified. See usage with option -h." << std::endl;
    std::exit(1);
  }
  
  if (exists_cmd_option(argv, argv + argc, "-s13")) {
    config.covariance(0, 2) = config.covariance(2, 0) = strtod(get_cmd_option(argv, argv + argc, "-s13"), NULL);
  } else {
    std::cout << "Argument for -s13 not specified. See usage with option -h." << std::endl;
    std::exit(1);
  }
  
  if (exists_cmd_option(argv, argv + argc, "-s22")) {
    config.covariance(1, 1) = strtod(get_cmd_option(argv, argv + argc, "-s22"), NULL);
    assert(("Variance of X_2 is positive", config.covariance(1, 1) > 0));
  } else {
    std::cout << "Argument for -s22 not specified. See usage with option -h." << std::endl;
    std::exit(1);
  }
  
  if (exists_cmd_option(argv, argv + argc, "-s23")) {
    config.covariance(1, 2) = config.covariance(2, 1) = strtod(get_cmd_option(argv, argv + argc, "-s23"), NULL);
  } else {
    std::cout << "Argument for -s23 not specified. See usage with option -h." << std::endl;
    std::exit(1);
  }
  
  if (exists_cmd_option(argv, argv + argc, "-s33")) {
    config.covariance(2, 2) = strtod(get_cmd_option(argv, argv + argc, "-s33"), NULL);
    assert(("Variance of X_3 is positive", config.covariance(2, 2) > 0));
  } else {
    std::cout << "Argument for -s33 not specified. See usage with option -h." << std::endl;
    std::exit(1);
  }
  
  std::cout << "Input covariance matrix:" << std::endl;
  std::cout << config.covariance << std::endl;


	void *context = &config;				// not required by MultiNest, any additional information user wants to pass


	// double C[2] = {0.5, 0.5};
	// double ll = -1.0;
	// 
	// LogLike(C, ndims, nPar, ll, context);
	// std::cout << ll << std::endl;
	// 
	// C[0] = 0.25;
	// LogLike(C, ndims, nPar, ll, context);
	// std::cout << ll << std::endl;
	// 
	// std::exit(0);
	
	// calling MultiNest
  nested::run(IS, mmodal, ceff, nlive, tol, efr, ndims, nPar, nClsPar, maxModes, updInt, Ztol, root, seed, pWrap, fb, resume, outfile, initMPI, logZero, maxiter, LogLike, dumper, context);

}

/***********************************************************************************************************************/


