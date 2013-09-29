#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> // getimeofday

#include "hmm.h"


#define IDX(i,j,d) (((i)*(d))+(j)) 
#define HANDLE_ERROR(msg) \
	  do { perror(msg); exit(EXIT_FAILURE); } while (0)


void print_hmm(Hmm *hmm)
{
	int j;
	int k;
	int n_st = hmm->nstates;
	int n_sy = hmm->nsymbols;

	printf("# state transition probability (A)\n");
	for (j = 0; j < n_st; j++) {
		for (k = 0; k < n_st; k++) {
			printf(" %.4f", hmm->a[(j * n_st) + k]);
		}
		printf("\n");
	}

	printf("# state output probility (B)\n");
	for (j = 0; j < n_sy; j++) {
		for (k = 0; k < n_st; k++) {
			printf(" %.4f", hmm->b[(j * n_st) + k]);
		}
		printf("\n");
	}

	printf("# initial state probability (Pi)\n");
	for (j = 0; j < n_st; j++) {
		printf(" %.4f", hmm->pi[j]);
	}
	printf("\n\n");
}


void print_obs(Obs *obs)
{
	int j;
	printf("Observation sequence:\n");
	for (j = 0; j < obs->length; j++) {
		printf(" %i", obs->data[j]);
	}
	printf("\n\n");
}


/* Free the memory associated with the HMM and observation sequence */
void free_vars(Hmm *hmm, Obs *obs)
{
	if (hmm != NULL) {
		if (hmm->a != NULL) {
			free(hmm->a);
		}
		if (hmm->b != NULL) {
			free(hmm->b);
		}
		if (hmm->pi != NULL) {
			free(hmm->pi);
		}
		free(hmm);
	}

	if (obs != NULL) {
		if (obs->data != NULL) {
			free(obs->data);
		}
		free(obs);
	}
}


void read_hmm_file(char *configfile, Hmm *hmm, Obs *obs, int transpose)
{
	FILE *fin, *bin;
	// Initialize line buffer
	char *linebuf = NULL;
	size_t buflen = 0;
	// Initialize parsing variables
	int c, k;
	float d;
	opterr = 0;
	// Initialize HMM parameters 
	int nStates = 0;            /* Number of possible states */
	int nSymbols = 0;           /* Number of possible symbols */
	int nSeq = 0;               /* Number of observation sequences */
	int length = 0;             /* Length of each observation sequence */
	// Initialize counter variables
	int i, j;

	// Open config file
	if (configfile == NULL) {
		fin = stdin;
	} else {
		fin = fopen(configfile, "r");
		if (fin == NULL) {
			HANDLE_ERROR("fopen");
		}
	}

	/* Parse the config file and extract HMM parameters */
	i = 0;
	while ((c = getline(&linebuf, &buflen, fin)) != -1) {

		/* Ignore blank and comment lines (start with '#') */
		if (c <= 1 || linebuf[0] == '#') {
			continue;
		}

		/* First section - read number of states */
		if (i == 0) {
			if (sscanf(linebuf, "%d", &nStates) != 1) {
				fprintf(stderr, "Config file format error: %d\n", i);
				free_vars(hmm, obs);
				exit(EXIT_FAILURE);
			}

			/* Allocate memory for Pi */
			hmm->nstates = nStates;
			hmm->pi = (float *) malloc(sizeof(float) * nStates);
			if (hmm->pi == NULL) {
				HANDLE_ERROR("malloc");
			}

			/* Allocate memory for A */
			hmm->a = (float *) malloc(sizeof(float) * nStates * nStates);
			if (hmm->a == NULL) {
				HANDLE_ERROR("malloc");
			}

			/* Second section - read number of possible symbols */
		} else if (i == 1) {
			if (sscanf(linebuf, "%d", &nSymbols) != 1) {
				fprintf(stderr, "Config file format error: %d\n", i);
				free_vars(hmm, obs);
				exit(EXIT_FAILURE);
			}

			/* Allocate memory for B */
			hmm->nsymbols = nSymbols;
			hmm->b = (float *) malloc(sizeof(float) * nStates * nSymbols);
			if (hmm->b == NULL) {
				HANDLE_ERROR("malloc");
			}

			/* Third section - read initial state probabilities (Pi) */
		} else if (i == 2) {

			/* Open memory stream and read in line */
			bin = fmemopen(linebuf, buflen, "r");
			if (bin == NULL) {
				HANDLE_ERROR("fmemopen");
			}

			/* Parse through memory stream and store probability values */
			for (j = 0; j < nStates; j++) {
				if (fscanf(bin, "%f", &d) != 1) {
					fprintf(stderr, "Config file format error: %d\n", i);
					free_vars(hmm, obs);
					exit(EXIT_FAILURE);
				}
				hmm->pi[j] = d;
			}
			fclose(bin);

			/* Fourth section - read in state transition probabilities (A) */
		} else if (i <= 2 + nStates) {

			/* Open memory stream and read in line */
			bin = fmemopen(linebuf, buflen, "r");
			if (bin == NULL) {
				HANDLE_ERROR("fmemopen");
			}

			/* Parse through memory stream and store probability values */
			for (j = 0; j < nStates; j++) {
				if (fscanf(bin, "%f", &d) != 1) {
					fprintf(stderr, "Config file format error: %d\n", i);
					free_vars(hmm, obs);
					exit(EXIT_FAILURE);
				}
				hmm->a[IDX((i - 3), j, nStates)] = d;
			}
			fclose(bin);

			/* Fifth section - read in output probabilities (B) */
		} else if ((i <= 2 + nStates * 2) && (transpose == 0)) {

			/* Open memory stream and read in line */
			bin = fmemopen(linebuf, buflen, "r");
			if (bin == NULL) {
				HANDLE_ERROR("fmemopen");
			}

			/* Parse through memory stream and store probability values */
			for (j = 0; j < nSymbols; j++) {
				if (fscanf(bin, "%f", &d) != 1) {
					fprintf(stderr, "Config file format error: %d\n", i);
					free_vars(hmm, obs);
					exit(EXIT_FAILURE);
				}
				hmm->b[j * nStates + (i - 3 - nStates)] = d;
			}
			fclose(bin);

			/* Fifth section (again) - read in output probabilities (B) if transposed */
		} else if ((i <= 2 + nStates + nSymbols) && (transpose)) {

			/* Open memory stream and read in line */
			bin = fmemopen(linebuf, buflen, "r");
			if (bin == NULL) {
				HANDLE_ERROR("fmemopen");
			}

			/* Parse through memory stream and store probability values */
			for (j = 0; j < nStates; j++) {
				if (fscanf(bin, "%f", &d) != 1) {
					fprintf(stderr, "Config file format error: %d\n", i);
					free_vars(hmm, obs);
					exit(EXIT_FAILURE);
				}
				hmm->b[(i - 3 - nStates) * nStates + j] = d;
			}
			fclose(bin);

			/* Sixth section - read in data size (seq, length) */
		} else if ((i == 3 + nStates * 2) && (transpose == 0)) {

			/* Read in data sequence and length values */
			if (sscanf(linebuf, "%d %d", &nSeq, &length) != 2) {
				fprintf(stderr, "Config file format error: %d\n", i);
				free_vars(hmm, obs);
				exit(EXIT_FAILURE);
			}
			obs->length = nSeq * length;

			/* Allocate memory for observation sequence */
			obs->data = (int *) malloc(sizeof(int) * nSeq * length);
			if (obs->data == NULL) {
				HANDLE_ERROR("malloc");
			}

			/* Sixth section (again) - read in data size if B is transposed */
		} else if ((i == 3 + nStates + nSymbols) && (transpose)) {

			/* Read in data sequence and length values */
			if (sscanf(linebuf, "%d %d", &nSeq, &length) != 2) {
				fprintf(stderr, "Config file format error: %d\n", i);
				free_vars(hmm, obs);
				exit(EXIT_FAILURE);
			}
			obs->length = nSeq * length;

			/* Allocate memory for observation sequence */
			obs->data = (int *) malloc(sizeof(int) * nSeq * length);
			if (obs->data == NULL) {
				HANDLE_ERROR("malloc");
			}

			/* Seventh section - read in observation sequence */
		} else if ((i <= 3 + nStates * 2 + nSeq) && (transpose == 0)) {

			/* Open memory stream and read in line */
			bin = fmemopen(linebuf, buflen, "r");
			if (bin == NULL) {
				HANDLE_ERROR("fmemopen");
			}

			/* Parse through observation sequence and store values (read down, then across) */
			for (j = 0; j < length; j++) {
				if (fscanf(bin, "%d", &k) != 1 || k < 0 || k >= nSymbols) {
					fprintf(stderr, "Config file format error: %d\n", i);
					free_vars(hmm, obs);
					exit(EXIT_FAILURE);
				}
				obs->data[j * nSeq + (i - 4 - nStates * 2)] = k;
			}
			fclose(bin);
		} else if ((i <= 3 + nStates + nSymbols + nSeq) && (transpose)) {

			/* Open memory stream and read in line */
			bin = fmemopen(linebuf, buflen, "r");
			if (bin == NULL) {
				HANDLE_ERROR("fmemopen");
			}

			/* Parse through observation sequence and store values (read down, then across) */
			for (j = 0; j < length; j++) {
				if (fscanf(bin, "%d", &k) != 1 || k < 0 || k >= nSymbols) {
					fprintf(stderr, "Config file format error: %d\n", i);
					free_vars(hmm, obs);
					exit(EXIT_FAILURE);
				}
				obs->data[j * nSeq + (i - 4 - (nStates + nSymbols))] = k;
			}
			fclose(bin);
		}

		/* Increment line counter */
		i++;

	}

	/* Close files and free memory */
	fclose(fin);
	if (linebuf) {
		free(linebuf);
	}

	/* Throw error if configuration file was not read properly */
	if (i < 4 + nStates * 2 + nSeq) {
		fprintf(stderr, "Configuration incomplete\n");
		free_vars(hmm, obs);
		exit(EXIT_FAILURE);
	}

}


int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;
	return (diff<0);
}


void tic(struct timeval *timer)
{
	gettimeofday(timer, NULL);
}


void toc(struct timeval *timer)
{
	struct timeval tv_end, tv_diff;
	gettimeofday(&tv_end, NULL);
	timeval_subtract(&tv_diff, &tv_end, timer);
	printf("%ld.%06ld\n", tv_diff.tv_sec, tv_diff.tv_usec);
}










