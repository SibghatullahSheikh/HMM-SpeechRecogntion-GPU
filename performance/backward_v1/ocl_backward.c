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
#include "run_opencl_backward.h"

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
		"../resources/config_32N32M_B.txt",
		"../resources/config_32N32M_A.txt",
		"../resources/config_32N32M_prior.txt",
		"../resources/config_64N64M_B.txt",
		"../resources/config_64N64M_A.txt",
		"../resources/config_64N64M_prior.txt",
		"../resources/config_128N128M_B.txt",
		"../resources/config_128N128M_A.txt",
		"../resources/config_128N128M_prior.txt",
		"../resources/config_256N256M_B.txt",
		"../resources/config_256N256M_A.txt",
		"../resources/config_256N256M_prior.txt",
		"../resources/config_512N512M_B.txt",
		"../resources/config_512N512M_A.txt",
		"../resources/config_512N512M_prior.txt",
		"../resources/config_1024N1024M_B.txt",
		"../resources/config_1024N1024M_A.txt",
		"../resources/config_1024N1024M_prior.txt",
	};

	// variables
	int i,j,k;
	int Len;
	int debug=0;

	int job=0;

	// select job frome commmand line
	int argi;
	if (argc == 1)
	{
		puts("Please specify an option.\nUsage: \"./ocl_fo -job number(0-5) \"\n");
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

	//printf("job = %d\n", job);
	if( job > 5) {
		printf("Job number exceeds the limit 5! Exit Programm!\n");
		exit(1);	
	}



	HMM *word;
	word = (HMM*)malloc(sizeof(HMM));

	Len = getLineNum(files[job*3+2]);	
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
		puts("pri");
		check_pri(word);	
	}


	//----------------------
	// run backward algorithm
	//----------------------

	//---------------------------
	// GPU Version
	//---------------------------

	run_opencl_backward(word);

	//---------------------------
	// CPU Version
	//---------------------------


	puts("\n=>CPU");

	struct timeval cpu_timer;

	int N = word->nstates;
	int T = word->len;
	float *B = word->b;
	float *A = word->a;

	double tmp, beta_sum;

	float *beta; // NxT
	beta = (float*)malloc(sizeof(float)*N*T);

	init_2d_f(beta,N,T,0.f);	

	for(i = 0 ; i < N ; ++i){
		beta[i*T + T-1] = 1.f;	
	}

	float *betaB;
	betaB = (float*)malloc(sizeof(float)*N);	

	// start timing
	tic(&cpu_timer);

	for(j = T-2; j >= 0; --j)
	{


		for(i = 0; i < N ; ++i)
		{ // go through each state
			betaB[i] = beta[i*T + j+1] * B[i*T + j+1];
		}

		beta_sum = 0.0;
		for(i = 0; i < N ; ++i) // row for A[]
		{
			tmp = 0.0;
			for( k = 0 ; k < N ; ++ k){ // col for A[]
				tmp += A[i*N + k]*	betaB[k];
			}
			beta[i*T + j] = (float)tmp;
			beta_sum += (float)tmp;
		}

		// scaling
		for(i=0;i<N;++i){			
			beta[i*T + j] /= (float)beta_sum;
		}

	}
	// end timing
	toc(&cpu_timer);


	// check out the beta
	//check_2d_f(beta,N,T);	

	// free memory

	free_hmm(word);
	free(beta);
	free(betaB);


	return 0;
}


