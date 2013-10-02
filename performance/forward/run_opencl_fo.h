#ifndef RUN_OPENCL_FO_H
#define RUN_OPENCL_FO_H

#define BUFSIZE 1000000

typedef struct{
	float *b; 	//NxT
	float *a; 	//NxN
	float *pri; //Nx1
	int len;
	int nstates;
} HMM;

void run_opencl_fo(HMM *word, int job);




// some useful functions
void read_config(HMM* word, char **files,int job, int states, int len);

int getLineNum(char *file);

void copy_2d(float **from_array, float **to_array, int row, int col);

void read_a(char *file , HMM* word, int row ,int col);

void read_b(char *file , HMM* word, int row ,int col);

void read_pri(char *file ,HMM* word, int len);

void check_a(HMM *word);

void check_b(HMM *word);

void check_pri(HMM *word);

void check_2d_f(float *x, int row, int col);

void free_hmm(HMM *word);

void start(struct timeval *timer);

void end(struct timeval *timer);

void timeval_diff(struct timeval *tdiff, struct timeval *t1, struct timeval *t2);

void transpose(float *A, float *A_t, int row, int col);




#endif
