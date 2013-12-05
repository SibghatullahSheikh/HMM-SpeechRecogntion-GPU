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

void check_3d_f(float ***array, unsigned int row, unsigned int col, unsigned int N)
{
	unsigned int i,j,k;

	for(k=0;k<N;++k)
	{
		printf("[][][%d] = \n",k);
		for(i=0;i<row;++i)
		{
			for(j=0;j<col;++j)
			{
				printf("%10.5f ",array[i][j][k]);	

			}
			printf("\n");
		}
	}

}

void check_2d_f(float **array, unsigned int row, unsigned int col)
{
	unsigned int i,j;

	for(i=0;i<row;++i)
	{
		for(j=0;j<col;++j)
		{
			printf("%10.5f ",array[i][j]);	

		}
		printf("\n");
	}

}


void check_1d_f(float *array, unsigned int len)
{
	unsigned int i;

	for(i=0;i<len;++i)
	{
		printf("%10.5f \n",array[i]);	
	}

}


void transpose(float **in, float **out, unsigned int row, unsigned int col)
{
	unsigned int i,j;
	for(i=0;i<row;++i)
	{
		for(j=0;j<col;++j)
		{
			out[j][i] = in[i][j];
		}
	}
}


void symmetrize_f(float **in, unsigned int N)
{
	unsigned int i,j;
	for(i=0;i<N;++i){
		for(j=0;j<i;++j){
			// compare (i,j) with (j,i)
			if(in[i][j]<in[j][i]){
				in[i][j] = in[j][i];
			}else{
				in[j][i] = in[i][j];
			}
		}
	}
}

void rand_1d_f(float *a, unsigned int N)
{
	int i;
	for(i=0;i<N;++i){
		a[i] = (float)rand()/(float)RAND_MAX;
	}

}


void rand_2d_f(float **a, unsigned int row, unsigned int col)
{
	//srand(time(NULL));
	int i,j;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			a[i][j] = (float)rand()/(float)RAND_MAX;
		}
	}

}

void randperm_1d(int *a, unsigned int N)
{
	// returns a vector containing a random permutation of the integers 0:N-1
	int i,p,tmp;
	for(i=0;i<N;++i){
		a[i]=i;
	}

	for(i=0;i<N;++i){
		p = rand()%N;
		//swap
		tmp = a[i];
		a[i] = a[p];
		a[p] = tmp;
	}

}

void normalise_1d_f(float *x, unsigned int len)
{
	unsigned int i;
	double sum=0.0;
	for(i=0;i<len;++i){
		sum += x[i];
	}

	if(sum ==0.0)
	{
		printf("Warning: sum along dimension is zero! Change it to 1. (FILE:%s,LINE:%d)\n",__FILE__,__LINE__);
		sum = 1.0;
	}

	for(i=0;i<len;++i){
		x[i] /= (float)sum;
	}

}


void normalise_2d_f(float **x, unsigned int row, unsigned int col)
{
	/*
	   if(nargin < 2)
	   z = sum(A(:));
	   z(z==0) = 1;
	   A = A./z;
	   else
	   z = sum(A, dim);
	   z(z==0) = 1;
	   A = bsxfun(@rdivide, A, z);
	   end
	 */

	// sum, columnwise
	unsigned int i,j;
	double sum=0.0;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			sum = sum + x[i][j];
		}
	}

	if(sum == 0.0)
	{
		printf("Warning: sum along dimension is zero! Change it to 1. (FILE:%s,LINE:%d)\n",__FILE__,__LINE__);
		sum = 1.0;
	}

	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			x[i][j] /= (float)sum;
		}
	}

}

void mk_stochastic_2d_f(float **x, unsigned int row, unsigned int col, float **out)
{
	//  Z = sum(T,2); 
	//  S = Z + (Z==0);

	unsigned int i,j;

	float **norm;  // row x col
	norm=(float **)malloc(sizeof(float*)*row);
	for(i=0;i<row;++i){
		norm[i] = (float*)malloc(sizeof(float)*col);	
	}

	float *Z;
	Z =(float*)malloc(sizeof(float)*row);

	float *Z1;
	Z1 =(float*)malloc(sizeof(float)*row);
	init_1d_f(Z1,row,0.f);

	double sum;
	// sum across row
	for(i=0;i<row;++i){
		sum=0.0;    
		for(j=0;j<col;++j){
			sum = sum + x[i][j];
		}
		Z[i] = (float)sum;
	}   

	for(i=0;i<row;++i){
		if(Z[i]==0){
			Z1[i]=1.0;  
		}
	}

	for(i=0;i<row;++i){
		Z1[i]=Z1[i]+Z[i];   
	}

	//  norm = repmat(S, 1, size(T,2));
	//  norm = [S S .. S]
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			norm[i][j] = Z1[i];  
		}
	}   

	//  T = T ./ norm;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			out[i][j] = x[i][j]/norm[i][j];    
		}
	}   

	// release
	for(i=0;i<row;++i){
		free(norm[i]);
	}
	free(norm);
	free(Z);
	free(Z1);
}

float norm_square(float **U, unsigned int j, unsigned int rows)
{
	double norm = 0.0;
	unsigned int i=0;
	for(i=0;i<rows;++i){
		norm += U[i][j]*U[i][j];
	}

	norm = sqrt(norm);

	return (float)norm;
}


void eye_2d_d(double **U,unsigned int row, unsigned int col)
{
	int i,j;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			U[i][j] = (i==j)? 1.0: 0.0;
		}
	}
}


void get2ddiag_d(double **from,double *to, unsigned int row, unsigned int col)
{
	if(row != col){
		puts("get2ddiag_f() currently works for square matrix only.");
		exit(1);
	}

	int i;
	int n = row;

	for(i=0;i<n;++i){
		to[i] =  from[i][i];
	}
}

void copy_2d_d(double **a, double **a_pre, int row, int col)
{
	int i,j;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			a_pre[i][j]  =  a[i][j];
		}
	}
}

void copy_1d_d(double *from, double *to, int len)
{
	int i;
	for(i=0;i<len;++i){
		to[i] = from[i];
	}
}

void init_1d_d(double *array,unsigned int len, double value)
{
	int i;
	for(i=0;i<len;++i){
		array[i] = value;
	}
}




