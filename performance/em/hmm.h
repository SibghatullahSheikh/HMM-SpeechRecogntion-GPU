#ifndef HMM_H
#define HMM_H

#define BUFSIZE 1000000

typedef struct{
	float *b; 	//NxT
	float *a; 	//NxN
	// float *pri; //Nx1
	float *alpha; //NxT
	float *beta; //NxT
	int len;
	int nstates;
} HMM;

// some useful functions
void read_config(HMM* word, char **files,int job, int states, int len);

int getLineNum(char *file);

void copy_2d(float **from_array, float **to_array, int row, int col);

void read_a(char *file , HMM* word, int row ,int col);

void read_b(char *file , HMM* word, int row ,int col);

void read_alpha(char *file , HMM* word, int row ,int col);

void read_beta(char *file , HMM* word, int row ,int col);

//void read_pri(char *file ,HMM* word, int len);

void check_a(HMM *word);

void check_b(HMM *word);

//void check_pri(HMM *word);

void check_2d_f(float *x, int row, int col);

void check_1d_f(float *x, int len);

void free_hmm(HMM *word);

void start(struct timeval *timer);

void end(struct timeval *timer);

void timeval_diff(struct timeval *tdiff, struct timeval *t1, struct timeval *t2);

void transpose(float *A, float *A_t, int row, int col);

void init_3d_f(float *frames, int row, int col, int height , float val);

void init_2d_f(float *frames, int row, int col, float val);

void init_1d_f(float *frames, int len, float val);

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);

void tic(struct timeval *timer);

void toc(struct timeval *timer);

#endif
