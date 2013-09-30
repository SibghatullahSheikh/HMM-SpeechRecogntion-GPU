#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
//#include <time.h>
#include <sys/time.h>


#define BUFSIZE 1000000

void read_config(char **files,int job, float **B, float **A, float *prior, int len);
int getLineNum(char *file);
void copy_2d(float **from_array, float **to_array, int row, int col);
void read_2d(char *file ,float **b, int row ,int col);
void check_2d_f(float **array, int row, int col);


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

	float **B;
	float **A;
	float *prior;


	
	for(job = 0 ; job < 6 ; ++job){
		printf("\n\njob (%d) => ", job);

		Len = getLineNum(files[job*3+2]);	
		printf("%d\n",Len);

		//read B,A,prior
		//puts("Read the following files.");
		//read_config(files,job,B,A,prior,Len);


	}





	return 0;
}

void read_config(char **files,int job, float **B, float **A, float *prior, int len)
{
	// variables
	int i,j;
	int N = len;

	// allocate parameters
	float **b;
	b = (float**)malloc(sizeof(float*)*N);
	for(i = 0; i < N; ++i) {
		b[i] = (float*)malloc(sizeof(float)*N);
	}

	float **a;
	a = (float**)malloc(sizeof(float*)*N);
	for(i = 0; i < N; ++i) {
		a[i] = (float*)malloc(sizeof(float)*N);
	}

	float *pri;
	pri = (float*)malloc(sizeof(float)*N);


	for(i = 0; i < 3; ++i){
		//printf("%s\n",files[job*3+i]);	

		// read B
		if(i == 0){
			printf("read B[%d][%d] \t\t", N, N);
			// read matrix
			read_2d(files[job*3+i],b,N,N);
			
			//if(job == 0){
			//	check_2d_f(b,N,N);	
			//}

			//copy_2d(b,B,N,N);

			printf("done!\n");
		}

		// read A
		if(i == 1){
			printf("read A[%d][%d] \t\t", N, N);
			printf("done!\n");
		}

		// read prior
		if(i == 2){
			printf("read prior[%d] \t\t", N);
			printf("done!\n");
		}


	}	


	// de-allocate
	free(pri);
	for(i=0;i<N;++i){
		free(a[i]);
		free(b[i]);
	}
	free(a);
	free(b);

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


void read_2d(char *file ,float **b, int row ,int col)
{
	int i;
	int lineNum = 0;
	char buf[BUFSIZE];
	char *pch;
	
	FILE *fr;
	fr = fopen(file, "rb");
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
				sscanf(pch,"%f", &b[lineNum][i]);
				i++;
				pch = strtok (NULL, " ,.-");


			}

			/*
			int i;
			for(i=0;i<col;++i){
				strcat(format,"%f ");
			}	

			sscanf(buf,format, );
			*/

			lineNum++;
		}
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


void check_2d_f(float **array, int row, int col)
{
	int i,j;
	for(i=0;i<row;++i)
	{
		for(j=0;j<col;++j)
		{
			printf("%10.4e ",array[i][j]);	

		}
		printf("\n");
	}

}

