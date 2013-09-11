#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"


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

void check_2d_f(float **array, unsigned int row, unsigned int col)
{
	unsigned int i,j;

	for(i=0;i<row;++i)
	{
		printf("row = %d\n", i);
		for(j=0;j<col;++j)
		{
			printf("%10.5f \n",array[i][j]);	
		
		}
		printf("\n");
	}

}


