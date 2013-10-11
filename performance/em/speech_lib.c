//-----------------------------------------------------------
//
//  Speech Utility Functions
//
//-----------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "speech_lib.h"
#include "hmm.h"

void normalise_1d_f(float *x, int len)
{
	int i;
	double sum=0.0;
	for(i=0;i<len;++i){
		sum += x[i];
	}

	if(sum == 0.0)
	{
		printf("Warning: sum along dimension is zero! Change it to 1. (FILE:%s,LINE:%d)\n",__FILE__,__LINE__);
		sum = 1.0;
	}

	for(i=0;i<len;++i){
		x[i] /= (float)sum;
	}

}


void normalise_2d_f(float *x, int row, int col)
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
	int i,j;
	double sum=0.0;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			sum = sum + x[i*col + j];
		}
	}

	if(sum == 0.0)
	{
		printf("Warning: sum along dimension is zero! Change it to 1. (FILE:%s,LINE:%d)\n",__FILE__,__LINE__);
		sum = 1.0;
	}

	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			x[i*col + j] /= (float)sum;
		}
	}

}


void mk_stochastic_2d_f(float *x, int row, int col, float *out)
{
	//  Z = sum(T,2); 
	//  S = Z + (Z==0);
	int i,j;

	float *norm;  // row x col
	norm=(float *)malloc(sizeof(float)*row*col);

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
			sum = sum + x[i*col + j];
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
			norm[i*col + j] = Z1[i];  
		}
	}   

	//  T = T ./ norm;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			out[i*col + j] = x[i*col + j]/norm[i*col + j];    
		}
	}   

	// release
	free(norm);
	free(Z);
	free(Z1);
}



