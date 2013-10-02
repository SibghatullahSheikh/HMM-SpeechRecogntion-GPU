#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/opencl.h>

#include "../../utils/ocl_utils.h"

#include "run_opencl_fo.h"

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
	int job;
	int Len;
	int debug=0;


	for(job = 0 ; job < 6 ; ++job){
		HMM *word;
		word = (HMM*)malloc(sizeof(HMM));

		printf("\n\njob (%d) => ", job);

		Len = getLineNum(files[job*3+2]);	
		printf("%d\n",Len);

		//read B,A,prior
		puts("Read the following files.");

		//read_config(files,job,B,A,prior,Len);
		read_config(word,files,job,Len,Len);

		puts("Done!");
		if( debug && job == 0 ) {
			puts("a");
			check_a(word);	
			puts("b");
			check_b(word);	
			puts("pri");
			check_pri(word);	
		}


		//----------------------
		// run forward algorithm
		//----------------------

		//---------------------------
		// GPU Version
		//---------------------------

		run_opencl_fo(word, job);


		//---------------------------
		// CPU Version
		//---------------------------

		puts("\nCPU");

		int N = word->nstates;
		int T = word->len;
		float *B = word->b;
		float *A = word->a;
		float *prior = word->pri;

		double tmp, alpha_sum;
		double log_likelihood;

		float *alpha; // NxT
		alpha = (float*)malloc(sizeof(float)*N*T);

		float *A_t; // NxN
		A_t = (float*)malloc(sizeof(float)*N*N);

		transpose(A, A_t, N, T);	

		log_likelihood = 0.0;

		for(j=0;j<T;++j)
		{
			alpha_sum = 0.0;

			if(j==0){ // initialize
				for(i=0;i<N;++i){
					alpha[i*T + 0] = B[i*T + 0] * prior[i];	
					alpha_sum += alpha[i*T + 0];
				}
			}else{ // move forward
				for(i=0;i<N;++i)
				{ // go through each state
					tmp = 0.0;	
					for(k=0;k<N;++k){
						tmp += A_t[i*N + k] * alpha[k*T + j-1];
					}

					alpha[i*T + j] = (float)tmp * B[i*T + j];
					alpha_sum += alpha[i*T + j];
				}
			}

			// scaling
			for(i=0;i<N;++i){			
				alpha[i*T + j] /= alpha_sum;
			}

			log_likelihood += log(alpha_sum);
		}


		printf("log_likelihood = %lf\n", log_likelihood);


		// free memory

		free_hmm(word);
		free(A_t);
		free(alpha);

	}

	return 0;
}


