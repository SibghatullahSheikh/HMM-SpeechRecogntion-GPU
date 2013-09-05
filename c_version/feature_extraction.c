#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "feature_extraction.h"

#define PI  M_PI    /* pi to machine precision, defined in math.h */
#define TWOPI   (2.0*PI)

unsigned int getLineNumber(char *fileName)
{
	unsigned int lineNum=0;
	int ch;
	FILE *fr = fopen(fileName, "rb");
	if(fr == NULL) {
		printf("Can't open %s!\n", fileName);
		exit(1);
	}else{
		while((ch=fgetc(fr))!= EOF){
			if(ch == '\n'){
				lineNum++;
			}
		}
		fclose(fr);
	}
	return lineNum;
}


void read_frames(char *file, float **frames, unsigned int col, unsigned int row)
{
	FILE *fr;
	fr = fopen(file,"rb");
	if(fr==NULL){
		printf("Error when reading %s\n", file);
		exit(1);
	}

	unsigned int i,j;
	int ret;
	float data;
	for(i=0;i<row;i++){
		for(j=0;j<col;j++){
			ret = fscanf(fr,"%f", &data);	
			if(ret == 0){
				printf("fscanf error! %s line:%d\n", __FILE__, __LINE__);
				exit(1);
			}
			frames[i][j] = data;
		}
	}

	fclose(fr);
}

void init_2d_f(float **frames, unsigned int row, unsigned int col, float val)
{
	unsigned int i,j;
	for(i=0;i<row;++i)
	{
		for (j=0;j<col;++j)
		{
			frames[i][j] = val;			
		}
	}
}

void init_1d_f(float *array, unsigned int N, float val)
{
	unsigned int i;
	for (i=0;i<N;++i)
	{
		array[i] = val;			
	}
}

void buffer(float *sound, float **frames, unsigned int N, unsigned int framesize, unsigned int overlap, unsigned int frameNum)
{
	//double *array_buf = (double*) calloc(frameNum*framesize,sizeof(double));
	unsigned int i,j;
	unsigned int start_pos;
	unsigned int offset=framesize-overlap;

	// at the begining of array, #(overlap) of zeros need to be appeneded before array[]
	// so the pudo array[] will have N+overlap elements
	for(i=0;i<frameNum;i++){
		start_pos=i*offset;
		for(j=0;j<framesize;j++){
			if(start_pos-overlap>=0 && (start_pos-overlap+j)<N){
				frames[j][i] = sound[start_pos-overlap+j];
			}
		}
	}

}

void hamming_f(float *window, unsigned int N)
{
	// Equation 
	// w(n) = 0.54 - 0.46cos(2*pi*n/N), where 0<=n<N
	if(N==1){
		window[0] = 0.f;
	}else{
		unsigned int i;
		for(i=0; i<N; ++i){
			window[i] = 0.54 - 0.46*cos((TWOPI*i)/(float)(N-1));
		}
	}

}
