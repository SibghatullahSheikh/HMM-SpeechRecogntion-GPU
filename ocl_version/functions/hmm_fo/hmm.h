#ifndef HMM_H
#define HMM_H

#ifndef TRUE
#define TRUE    1
#endif 

#ifndef FALSE
#define FALSE   0
#endif

typedef struct {
	int nstates;   
	int nsymbols; 
	float *a;    
	float *b;   
	float *pi; 
}Hmm;


typedef struct {
	int length;             /**< the length of the observation sequence */
	int *data;              /**< the observation sequence */
} Obs;


void set_hmm_a(float *a);

void set_hmm_b(float *b);

void set_hmm_pi(float *pi);

void print_hmm(Hmm *hmm);

void print_obs(Obs *obs);

void free_vars(Hmm *hmm, Obs *obs);


// test functions
void read_hmm_file(char *configfile, Hmm *hmm, Obs *obs, int transpose);

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);

void tic(struct timeval *timer);

void toc(struct timeval *timer);





#endif
