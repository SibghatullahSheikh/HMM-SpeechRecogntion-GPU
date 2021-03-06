#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define BUFSIZE 1000000

typedef struct{
	float **b;
	float **a;
	float *pri;
	int len;
	int nstates;
} HMM;


//void read_config(char **files,int job, float **b, float **a, float *pri, int len);
void read_config(HMM* word, char **files,int job,int len);

int getLineNum(char *file);

void copy_2d(float **from_array, float **to_array, int row, int col);

void read_a(char *file , HMM* word, int row ,int col);

void read_b(char *file , HMM* word, int row ,int col);

void read_pri(char *file ,HMM* word, int len);

void check_a(HMM *word);

void check_b(HMM *word);

void check_pri(HMM *word);

void check_2d_f(float **x, int row, int col);

void free_hmm(HMM *word);

void start(struct timeval *timer);

void end(struct timeval *timer);

void timeval_diff(struct timeval *tdiff, struct timeval *t1, struct timeval *t2);

void transpose(float **A, float **A_t, int N);




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
	int job;
	int Len;
	int debug=0;
	
	struct timeval timer;



	
	for(job = 0 ; job < 6 ; ++job){
		HMM *word;
		word = (HMM*)malloc(sizeof(HMM));

		printf("\n\njob (%d) => ", job);

		Len = getLineNum(files[job*3+2]);	
		printf("%d\n",Len);

		//read B,A,prior
		puts("Read the following files.");

		//read_config(files,job,B,A,prior,Len);
		read_config(word,files,job,Len);

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
		int i,j,k;
		int N = word->nstates;
		int T = word->len;
		float **B = word->b;
		float **A = word->a;
		float *prior = word->pri;

		double tmp, alpha_sum;
		double log_likelihood;

		float **alpha;
		alpha = (float**)malloc(sizeof(float*)*N);
		for(i=0;i<N;++i){
			alpha[i] = (float*)malloc(sizeof(float)*T);
		}
		


		float **A_t;
		A_t = (float**)malloc(sizeof(float*)*N);
		for(i=0;i<N;++i){
			A_t[i] = (float*)malloc(sizeof(float)*N);
		}

		transpose(A, A_t, N);
		if(job == 0 && debug){
			puts("Transpose A");
			check_2d_f(A_t,N,N);
		}
		
	
		// start Foward
		start(&timer);
		
		log_likelihood = 0.0;
		for(j=0;j<T;++j)
		{
			alpha_sum = 0.0;

			if(j==0){ // initialize
				for(i=0;i<N;++i){
					alpha[i][0] = B[i][0] * prior[i];	
					alpha_sum += alpha[i][0];
				}
			}else{ // move forward
				for(i=0;i<N;++i)
				{ // go through each state
					tmp = 0.0;	
					for(k=0;k<N;++k){
						tmp += A_t[i][k] * alpha[k][j-1];
					}

					alpha[i][j] = (float)tmp * B[i][j];
					alpha_sum += alpha[i][j];
				}
			}

			// scaling
			for(i=0;i<N;++i){			
				alpha[i][j] /= alpha_sum;
			}

			log_likelihood += log(alpha_sum);
		}

		end(&timer);
		// end Forward

		printf("log_likelihood = %lf\n", log_likelihood);



		// free memory
		free_hmm(word);
		for(i=0;i<N;++i){
			free(A_t[i]);
			free(alpha[i]);
		}
		free(A_t);
		free(alpha);

	}


	return 0;
}


//-----------------------------------------------------------
//
//	Utility Functions
//
//-----------------------------------------------------------

void read_config(HMM* word, char **files,int job, int len)
{
	// variables
	int i;
	int N = len;

	word->nstates = N;
	word->len = N;


	// allocate parameters
	//float **b = word->b;
	word->b = (float**)malloc(sizeof(float*)*N);
	for(i = 0; i < N; ++i) {
		word->b[i] = (float*)malloc(sizeof(float)*N);
	}

	//float **a = word->a;
	word->a = (float**)malloc(sizeof(float*)*N);
	for(i = 0; i < N; ++i) {
		word->a[i] = (float*)malloc(sizeof(float)*N);
	}

	//float *pri = word->pri;
	word->pri = (float*)malloc(sizeof(float)*N);


	for(i = 0; i < 3; ++i){
		//printf("%s\n",files[job*3+i]);	

		// read B
		if(i == 0){
			printf("read B[%d][%d] \t\t", N, N);
			read_b(files[job*3],word,N,N);
			printf("done!\n");
		}

		// read A
		if(i == 1){
			printf("read A[%d][%d] \t\t", N, N);
			read_a(files[job*3+1],word,N,N);
			printf("done!\n");
		}

		// read prior
		if(i == 2){
			printf("read prior[%d] \t\t", N);
			read_pri(files[job*3+2],word,N);
			printf("done!\n");
		}

	}	

}


int getLineNum(char *file)
{
	int lineNum=0;
	char buffer[BUFSIZE];
	FILE *fr = fopen(file, "rb");
	if(fr == NULL) {
		printf("Can't open %s!\n", file);
		exit(1);
	}else{
		while(fgets(buffer, sizeof(buffer), fr))
		{
			lineNum++;
		}
		fclose(fr);
	}

	return lineNum;
}


void read_pri(char *file ,HMM* word, int len)
{
	int lineNum = 0;
	char buf[100];
	
	FILE *fr;
	fr = fopen(file, "r");
	if(fr == NULL) {
		printf("Can't open %s!\n", file);
		exit(1);
	}else{
		while(fgets(buf, sizeof(buf), fr) != NULL)
		{
			sscanf(buf,"%f", &word->pri[lineNum]);
			lineNum++;
		}
		fclose(fr);
	}
}



void read_b(char *file , HMM* word, int row ,int col)
{
	int i;
	int lineNum = 0;
	char buf[BUFSIZE];
	char *pch;
	
	FILE *fr;
	fr = fopen(file, "r");
	if(fr == NULL) {
		printf("Can't open %s!\n", file);
		exit(1);
	}else{
		while(fgets(buf, BUFSIZE, fr) != NULL)
		{
			i = 0;
			pch =  strtok(buf," ");
			while(pch != NULL)
			{
				sscanf(pch,"%f", &word->b[lineNum][i]);
				i++;
				pch = strtok (NULL, " ");
			}

			lineNum++;
		}
		//printf("%d \n",lineNum);
		fclose(fr);
	}

}


void read_a(char *file , HMM* word, int row ,int col)
{
	int i;
	int lineNum = 0;
	char buf[BUFSIZE];
	char *pch;
	
	FILE *fr;
	fr = fopen(file, "r");
	if(fr == NULL) {
		printf("Can't open %s!\n", file);
		exit(1);
	}else{
		while(fgets(buf, BUFSIZE, fr) != NULL)
		{
			i = 0;
			pch =  strtok(buf," ");
			while(pch != NULL)
			{
				sscanf(pch,"%f", &word->a[lineNum][i]);
				i++;
				pch = strtok (NULL, " ");
			}

			lineNum++;
		}
		//printf("%d \n",lineNum);
		fclose(fr);
	}

}


void copy_2d(float **from_array, float **to_array, int row, int col)
{
	int i,j;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			to_array[i][j] = from_array[i][j];
		}	
	}

}


void check_pri(HMM *word)
{
	int i;
	int len = word->len;

	for(i=0;i<len;++i)
	{
		printf("%10.4e\n", word->pri[i]);	
	}
	printf("\n");

}


void check_a(HMM *word)
{
	int i,j;
	int row = word->nstates;
	int col = word->len;

	for(i=0;i<row;++i)
	{
		printf("row(%d)\n",i);
		for(j=0;j<col;++j)
		{
			printf("%10.4e\n",word->a[i][j]);	

		}
		printf("\n\n");
	}

}


void check_b(HMM *word)
{
	int i,j;
	int row = word->nstates;
	int col = word->len;

	for(i=0;i<row;++i)
	{
		printf("row(%d)\n",i);
		for(j=0;j<col;++j)
		{
			printf("%10.4e\n",word->b[i][j]);	

		}
		printf("\n\n");
	}

}



void free_hmm(HMM *word)
{
	int T,i;
	T = word->len;
	
	for(i = 0; i < T ; ++i) {
		free(word->a[i]);
		free(word->b[i]);
	}

	free(word->a);
	free(word->b);
	free(word->pri);

}

void start(struct timeval *timer)
{
	gettimeofday(timer, NULL);
}


void end(struct timeval *timer)
{
	struct timeval tend, tdiff;
	gettimeofday(&tend, NULL);
	timeval_diff(&tdiff, timer, &tend);
	printf("Elapsed Time = %ld.%06ld(s)\n",tdiff.tv_sec, tdiff.tv_usec);
}

void timeval_diff(struct timeval *tdiff, struct timeval *t1, struct timeval *t2)
{
	long int diff = (t2->tv_usec + 1000000*t2->tv_sec) - (t1->tv_usec + 1000000*t1->tv_sec);
	tdiff->tv_sec  = diff/1000000;
	tdiff->tv_usec = diff%1000000;
}


void transpose(float **A, float **A_t, int N)
{
	int i,j;
	for(i=0;i<N;++i){
		for(j=0;j<N;++j){
			A_t[j][i] = A[i][j];
		}	
	}

}

void check_2d_f(float **x, int row, int col)
{
	int i,j;

	for(i=0;i<row;++i)
	{
		printf("row(%d)\n",i);
		for(j=0;j<col;++j)
		{
			printf("%10.4e\n",x[i][j]);	

		}
		printf("\n\n");
	}
}

