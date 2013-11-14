#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdarg.h>

#include <CL/opencl.h>

#include "../../utils/ocl_utils.h"

#include "hmm.h"
#include "speech_lib.h"
#include "run_opencl_em.h"

void fatal(const char *fmt, ...)
{
	va_list va;
	va_start(va, fmt);
	fprintf(stderr, "fatal: ");
	vfprintf(stderr, fmt, va);
	fprintf(stderr, "\n");
	fflush(NULL);
	exit(1);
}

static void need_argument(int argc, char *argv[], int argi)
{
	if (argi == argc -1)
		fatal("option %s requires one argument.\n", argv[argi]);
}

//-----------------------------------------------------------
//
// Main Function	
//
//-----------------------------------------------------------

int main(int argc, char*argv[])
{
	// 6 config, each has three files to read
	char *files[] = {
		"resources/config_32N32M_B.txt",
		"resources/config_32N32M_A.txt",
		"resources/config_32N32M_alpha.txt",
		"resources/config_32N32M_beta.txt",
		"resources/config_32N32M_obs.txt",
		"resources/config_64N64M_B.txt",
		"resources/config_64N64M_A.txt",
		"resources/config_64N64M_alpha.txt",
		"resources/config_64N64M_beta.txt",
		"resources/config_64N64M_obs.txt",
		"resources/config_128N128M_B.txt",
		"resources/config_128N128M_A.txt",
		"resources/config_128N128M_alpha.txt",
		"resources/config_128N128M_beta.txt",
		"resources/config_128N128M_obs.txt",
		"resources/config_256N256M_B.txt",
		"resources/config_256N256M_A.txt",
		"resources/config_256N256M_alpha.txt",
		"resources/config_256N256M_beta.txt",
		"resources/config_256N256M_obs.txt",
		"resources/config_512N512M_B.txt",
		"resources/config_512N512M_A.txt",
		"resources/config_512N512M_alpha.txt",
		"resources/config_512N512M_beta.txt",
		"resources/config_512N512M_obs.txt",
		"resources/config_1024N1024M_B.txt",
		"resources/config_1024N1024M_A.txt",
		"resources/config_1024N1024M_alpha.txt",
		"resources/config_1024N1024M_beta.txt",
		"resources/config_1024N1024M_obs.txt",
	};


	// vables
	int i,j,n,k;
	int Len;

	int job=0;

	// select job frome commmand line
	int argi;
	if (argc == 1)
	{
		puts("Please specify an option.\nUsage: \"./ocl_em -job number(0-5) \"\n");
		exit(1);
	}

	for (argi = 1; argi < argc; ++argi)
	{
		if (!strcmp(argv[argi], "-job"))
		{
			need_argument(argc, argv,argi);
			job = atoi(argv[++argi]) ;
			continue;
		}

		if (argv[argi][0] == '-')
		{
			fatal("'%s' is not a valid command-line option.\n",argv[argi]);
		}
	}

	HMM *word;
	word = (HMM*)malloc(sizeof(HMM));

	//Len = getLineNum(files[job*3+2]);	
	if (job == 0)
	{
		Len = 32;	
	} 
	else if (job == 1) 
	{
		Len = 64;	
	} 
	else if (job == 2)
	{
		Len = 128;	
	} 
	else if (job == 3)
	{
		Len = 256;	
	} 
	else if (job == 4)
	{
		Len = 512;	
	} 
	else if (job == 5)
	{
		Len = 1024;	
	} 
	else 
	{
		printf("Job number exceeds the limit 5! Exit Programm!\n");
		exit(1);	
	}

	printf("config_%dN_%dM\n",Len, Len);

	//read B,A,prior
	printf("Read the following files...");

	//read_config(files,job,B,A,prior,Len);
	read_config(word,files,job,Len,Len,Len);



	//----------------------
	// run backward algorithm
	//----------------------

	//---------------------------
	// GPU Version
	//---------------------------

	 run_opencl_em(word);



	//------------------------------------------------
	// CPU Version
	//------------------------------------------------

	puts("\n=>CPU");
/*
	struct timeval cpu_timer;

	int N = word->nstates;
	int T = word->len;
	int D = word->features;

	float *B = word->b;
	float *A = word->a;
	float *alpha = word->alpha;
	float *beta  = word->beta;
	float *observations = word->obs; // D x T

	float *observations_t = (float*)malloc(sizeof(float)*T*D);

	transpose(observations,observations_t,D,T);



	// check_2d_f(B,N,T);
	// check_2d_f(A,N,N);
	// check_2d_f(Alpha,N,T);
	// check_2d_f(Beta,N,T);
	// check_2d_f(observations,D,T);


	// NxN
	float *xi_sum	= (float*)malloc(sizeof(float)*N*N);
	init_2d_f(xi_sum,N,N,0.f);

	// NxT
	float *gamma= (float*)malloc(sizeof(float)*N*T);
	init_2d_f(gamma,N,T,0.f);


	// intermediate data array
	float *beta_B 		  = (float*)malloc(sizeof(float)*N);
	float *alpha_beta     = (float*)malloc(sizeof(float)*N); // Nx1
	float *alpha_beta_B   = (float*)malloc(sizeof(float)*N*N); // NxN
	float *A_alpha_beta_B = (float*)malloc(sizeof(float)*N*N); // NxN

	int window;
	double tmp;

	// start timing
	tic(&cpu_timer);

	//------------------------------------//
	//	E-step
	//------------------------------------//
	for(window=0; window < (T-1) ; ++window)
	{
		// beta * B ( window +1 )
		for(j=0;j<N;++j)
		{
			beta_B[j] = beta[j*T + window + 1] * B[j*T + window + 1];
		} 

		if (window == 0 && 0){
			check_1d_f(beta_B,N);
		}

	
		// alpha * beta * B
		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				alpha_beta_B[j*N + k]= alpha[j*T + window] * beta_B[k];			
			}
		}

		if (window == 0 && 0){
			check_2d_f(alpha_beta_B,N,N);
		}


		// A * alpha * beta * B
		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				A_alpha_beta_B[j*N + k] = A[j*N + k] * alpha_beta_B[j*N + k];
			}
		}

		if (window == 0 && 0){
			check_2d_f(A_alpha_beta_B,N,N);
		}

		// normalise
		normalise_2d_f(A_alpha_beta_B,N,N);

		if (window == 0 && 0){
			check_2d_f(A_alpha_beta_B,N,N);
		}

		// xi_sum + A_alpha_beta_B
		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				xi_sum[j*N + k] += A_alpha_beta_B[j*N + k];
			}
		}

		if (window == 0 && 0){
			check_2d_f(xi_sum,N,N);
		}

		// alpha * beta ( window )
		for(j=0;j<N;++j)
		{
			alpha_beta[j] = alpha[j*T + window ] * beta [j*T + window ];
		} 


		// normalise
		normalise_1d_f(alpha_beta,N);

		// update gamma
		for(j=0;j<N;++j){
			gamma[j*T + window]= alpha_beta[j];
		} 


	}

	// update gamma for the last window (T -1)
	for(j=0;j<N;++j)
	{
		alpha_beta[j] = alpha[j*T + T-1] * beta [j*T + T-1];
	} 
	normalise_1d_f(alpha_beta,N);
	for(j=0;j<N;++j){
		gamma[j*T + T -1]= alpha_beta[j];
	} 

	//check_2d_f(gamma,N,T);




	//------------------------------------//
	//	M-step
	//------------------------------------//

	// expected prior
	float *exp_prior=(float*)malloc(sizeof(float)*N); // Nx1
	for(i=0;i<N;++i){
		exp_prior[i] = gamma[i*T + 0];
	}

	// expected A 
	float *exp_A=(float*)malloc(sizeof(float)*N*N);
	mk_stochastic_2d_f(xi_sum,N,N,exp_A);

	// expected sigma : DxDxN
	float *exp_sigma;
	exp_sigma = (float*)malloc(sizeof(float*)*D*D*N); // row
	init_3d_f(exp_sigma,D,D,N,0.f);

	// exp_mu : DxN
	float *exp_mu=(float*)malloc(sizeof(float)*D*N);
	init_2d_f(exp_mu,D,N,0.f);

	float *gamma_state_sum	= (float*)malloc(sizeof(float)*N); // Nx1

	for(i=0;i<N;++i){
		tmp = 0.0;
		for(j=0;j<T;++j)
		{
			tmp += gamma[i*T + j];
		}
		//Set any zeroes to one before dividing to avoid NaN
		if(tmp == 0.0){
			tmp = 1.0;
		}
		gamma_state_sum[i]=(float)tmp;
	}	


	// intermediate arrays
	float *gamma_obs 		= (float*)malloc(sizeof(float)*D*T); // DxT
	float *gamma_obs_mul	= (float*)malloc(sizeof(float)*D*D); // DxD
	float *exp_mu_mul		= (float*)malloc(sizeof(float)*D*D); // DxD
	float *gammaob_ob_t	    = (float*)malloc(sizeof(float)*D*D); // DxD




	//-----------------------------------------------------
	// need to configure observations/observations_t
	// need to define D (feature dimension)
	//-----------------------------------------------------

	// go through each state
	for(k=0;k<N;++k)
	{

		for(i=0;i<D;++i){ // num of repeats
			for(j=0;j<T;++j){
				gamma_obs[i*T + j]=gamma[k*T + j];	
			}	
		}

		for(i=0;i<D;++i){
			for(j=0;j<T;++j){
				gamma_obs[i*T + j] *= observations[i*T + j];	
			}
		}


		for(i=0;i<D;++i){
			tmp=0.0;
			for(j=0;j<T;++j){
				tmp += gamma_obs[i*T + j];	
			}
			exp_mu[i*N + k] = (float)tmp/gamma_state_sum[k]; 
		}

		// gamma_obs(DxT)  *  observations_t (TxD)
		for(i=0;i<D;++i){ // row
			for(j=0;j<D;++j){ // col
				tmp = 0.0;	
				for(n=0;n<T;++n){
					tmp += gamma_obs[i*T + n]*observations_t[n*D + j];
				}
				gammaob_ob_t[i*D + j] = (float)tmp;
			}
		}

		// opt : merge with previous step
		for(i=0;i<D;++i){ // row
			for(j=0;j<D;++j){ // col
				gammaob_ob_t[i*D + j] /= gamma_state_sum[k];
			}
		}


		// matrix multiplication
		for(i=0;i<D;++i){// row
			for(j=0;j<D;++j){ // col
				exp_mu_mul[i*D + j] = exp_mu[i*N + k] * exp_mu[j*N + k];
			}
		}

		// opt : merge with previous one
		// gammaob_bo_t  - exp_mu_mul
		for(i=0;i<D;++i){// row
			for(j=0;j<D;++j){ // col
				gammaob_ob_t[i*D + j] -= exp_mu_mul[i*D + j];
			}
		}

		// symmetrize()	
		symmetrize_f(gammaob_ob_t,D);

		// save to exp_sigma
		for(i=0;i<D;++i){// row
			for(j=0;j<D;++j){ // col
				exp_sigma[(i*D+j)*D + k] = gammaob_ob_t[i*D+j];
			}
		}

	}

	// % Ninja trick to ensure positive semidefiniteness
	for(k=0;k<N;++k)
	{
		for(i=0;i<D;++i)
		{
			for(j=0;j<D;++j)
			{
				if(i==j)	exp_sigma[(i*D+j)*D + k] += 0.01;
			}
		}
	}


	// need to update HMM parameters


	// end timing
	toc(&cpu_timer);




	// check out the beta
	//check_2d_f(beta,N,T);	

	//check_1d_f(exp_prior,N);	
	//check_2d_f(exp_A,N,N);	
	//check_2d_f(exp_mu,D,N);	

	// check exp_sigma

//	puts("sigma ");
//	for(i=0;i<D;++i)
//	{
//		printf("row %d\n", i);
//		for(j=0;j<D;++j)
//		{
//			printf("%.4e\n", exp_sigma[(i*D+j)*D + 1]);
//
//		}
//	}
//	printf("\n");

	free(xi_sum);
	free(gamma);
	free(beta_B);
	free(alpha_beta_B);
	free(alpha_beta);
	free(A_alpha_beta_B);

	free(exp_prior);
	free(exp_A);
	free(exp_mu);
	free(exp_sigma);
	free(gamma_state_sum);

	free(gamma_obs);
	free(observations_t);

	free(gamma_obs_mul);
	free(exp_mu_mul);
	free(gammaob_ob_t);
*/

	// free memory
	free_hmm(word);

	return 0;
}


