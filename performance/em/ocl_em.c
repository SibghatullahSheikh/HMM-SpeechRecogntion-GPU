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
		"resources/config_64N64M_B.txt",
		"resources/config_64N64M_A.txt",
		"resources/config_64N64M_alpha.txt",
		"resources/config_64N64M_beta.txt",
		"resources/config_128N128M_B.txt",
		"resources/config_128N128M_A.txt",
		"resources/config_128N128M_alpha.txt",
		"resources/config_128N128M_beta.txt",
		"resources/config_256N256M_B.txt",
		"resources/config_256N256M_A.txt",
		"resources/config_256N256M_alpha.txt",
		"resources/config_256N256M_beta.txt",
		"resources/config_512N512M_B.txt",
		"resources/config_512N512M_A.txt",
		"resources/config_512N512M_alpha.txt",
		"resources/config_512N512M_beta.txt",
		"resources/config_1024N1024M_B.txt",
		"resources/config_1024N1024M_A.txt",
		"resources/config_1024N1024M_alpha.txt",
		"resources/config_1024N1024M_beta.txt",
	};

	// vables
	int i,j,k;
	int Len;
	int debug=0;

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
	read_config(word,files,job,Len,Len);

	printf("Done!\n");
	if( debug && job == 0 ) {
		puts("a");
		check_a(word);	
		puts("b");
		check_b(word);	
		//puts("pri");
		//check_pri(word);	
	}


	//----------------------
	// run backward algorithm
	//----------------------

	//---------------------------
	// GPU Version
	//---------------------------

	//	run_opencl_backward(word);






	//------------------------------------------------
	// CPU Version
	//------------------------------------------------

	puts("\n=>CPU");

	struct timeval cpu_timer;

	int N = word->nstates;
	int T = word->len;
	float *B = word->b;
	float *A = word->a;
	float *alpha = word->alpha;
	float *beta  = word->beta;

	//check_2d_f(B,N,T);
	//check_2d_f(A,N,N);
	//check_2d_f(Alpha,N,T);
	//check_2d_f(Beta,N,T);

	// NxN
	float *xi_sum	= (float*)malloc(sizeof(float)*N*N);
	init_2d_f(xi_sum,N,N,0.f);

	// NxT
	float *gamma= (float*)malloc(sizeof(float)*N*N);
	init_2d_f(gamma,N,T,0.f);


	// intermediate data array
	float *beta_B = (float*)malloc(sizeof(float)*N);
	float *alpha_beta_B   = (float*)malloc(sizeof(float)*N*N); // NxN
	float *A_alpha_beta_B = (float*)malloc(sizeof(float)*N*N); // NxN

	int window;

	// start timing
	tic(&cpu_timer);


	for(window=0; window<T-1; ++window)
	{
		// beta * B ( window +1 )
		for(j=0;j<N;++j)
		{
			beta_B[j] = beta[j*T + window + 1] * B[j*T + window + 1];
		} 
	
		// alpha * beta * B
		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				alpha_beta_B[j*N + k]= alpha[j*T + window]*beta_B[k];			
			}
		}

		// A * alpha * beta * B
		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				A_alpha_beta_B[j*N + k] = A[j*N + k]*alpha_beta_B[j*N + k];
			}
		}

		// normalise
		normalise_2d_f(A_alpha_beta_B,N,N);


		// xi_sum + A_alpha_beta_B
		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				xi_sum[j*N + k] += A_alpha_beta_B[j*N + k];
			}
		}

		// beta * B ( window )
		for(j=0;j<N;++j)
		{
			beta_B[j] = beta[j*T + window ] * B[j*T + window ];
		} 

		// normalise
		normalise_1d_f(beta_B,N);

		// update gamma
		for(j=0;j<N;++j){
			gamma[j*T + window]= beta_B[j];
		} 


	}

	// update gamma for the last window (T -1)
	for(j=0;j<N;++j)
	{
		beta_B[j] = beta[j*T + T-1] * B[j*T + T-1];
	} 
	normalise_1d_f(beta_B,N);
	for(j=0;j<N;++j){
		gamma[j*T + T -1]= beta_B[j];
	} 

	// end timing
	toc(&cpu_timer);


	// check out the beta
	//check_2d_f(beta,N,T);	




	// free memory
	free_hmm(word);

	free(xi_sum);
	free(gamma);
	free(beta_B);
	free(alpha_beta_B);
	free(A_alpha_beta_B);

	return 0;
}


