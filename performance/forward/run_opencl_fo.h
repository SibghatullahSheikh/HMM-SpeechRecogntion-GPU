#ifndef RUN_OPENCL_FO_H
#define RUN_OPENCL_FO_H

typedef struct{
	float *b; 	//NxT
	float *a; 	//NxN
	float *pri; //Nx1
	int len;
	int nstates;
} HMM;

void run_opencl_fo(HMM *word);

#endif
