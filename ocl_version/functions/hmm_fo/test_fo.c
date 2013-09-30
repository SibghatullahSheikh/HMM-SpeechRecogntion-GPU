#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h> // getopt

#include "hmm.c"
//#include "hmm_fo.c"

//#include "../main/hmm_test.h"
//#include "../main/hmm_fo_cublas.h"

/* Read in HMM and observation from file and test FO */
int main(int argc, char *argv[]) {

	int c;
	Hmm *in_hmm;                /* Initial HMM */
	Hmm *out_hmm;               /* Output HMM */
	Obs *in_obs;                /* Observation sequence */
	char *configfile = NULL;
	float log_lik;              /* Output likelihood of FO */
	struct timeval timer;

	in_hmm = (Hmm *)malloc(sizeof(Hmm));
	out_hmm = (Hmm *)malloc(sizeof(Hmm));
	in_obs = (Obs *)malloc(sizeof(Obs));

	in_hmm->nstates = 2;
	in_hmm->nsymbols = 4;

	in_hmm->a = (float *)malloc(sizeof(float)*2*2);
	in_hmm->b = (float *)malloc(sizeof(float)*2*10); 
/*

	printf("\n");
	printf("HMM parameters read from file:\n");
	print_hmm(in_hmm);


	tic(&timer);
	log_lik = run_hmm_fo(in_hmm, in_obs);
	toc(&timer);
	printf("Log likelihood: %0.4f\n", log_lik);
*/


	free_vars(in_hmm, in_obs);



	return 0;
}

