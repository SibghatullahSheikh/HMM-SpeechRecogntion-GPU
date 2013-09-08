#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "train.h"

#define PI  M_PI    /* pi to machine precision, defined in math.h */
#define TWOPI   (2.0*PI)

void Train(float **observations, unsigned int featureDim, unsigned int frameNum, unsigned int hiddenstates)
{
	unsigned int i,j,k;
	unsigned int D = featureDim;
	unsigned int T = frameNum; 
	unsigned int N = hiddenstates; // hidden states

	// %prior = normalise(rand(N, 1));
	float *prior = (float*) malloc(sizeof(float)*N);
	prior[0] = 0.555826367548231;
	prior[1] = 0.384816327750085;
	prior[2] = 0.0593573047016839;

	//%A = mk_stochastic(rand(N));
	float **A = (float**) malloc(sizeof(float*)*N);
	for(i=0;i<N;++i)
	{
		A[i] = (float*) malloc(sizeof(float)*N);
	}

	A[0][0]=0.126558246942481;
	A[0][1]=0.438475341086527;
	A[0][2]=0.434966411970992;
	A[1][0]=0.459614415425850;
	A[1][1]=0.132462410695470;
	A[1][2]=0.407923173878680;
	A[2][0]=0.350943345583750;
	A[2][1]=0.355739578481455;
	A[2][2]=0.293317075934795;

	// Sigma = repmat(diag(diag(cov(observations'))), [1 1 N]);
	// perf hint: directly compute the diag diag cov 
	float *diag_diag_cov;
	diag_diag_cov = cov(observations,T,D); // 6x6
	for(i=0;i<D;i++){
		for(j=0;j<D;j++){
			if(i!=j){
				diag_diag_cov[i*D+j]=0;	
			}
		}
	}		   

	float *Sigma;
	Sigma = repmat(diag_diag_cov,D,D,N); // repeat (DxD) N times

	float *mu=(float*)malloc(sizeof(float)*N*D);// NxD = 3x6
	mu[0]=875;		mu[1]=562.5;	mu[2]=187.5;	mu[3]=1375;		mu[4]=1187.5;	mu[5]=1625;
	mu[6]=1000;		mu[7]=1562.5;	mu[8]=750;		mu[9]=875;		mu[10]=1437.5;	mu[11]=1187.5;
	mu[12]=687.5;	mu[13]=875;		mu[14]=562.5;	mu[15]=1562.5;	mu[16]=1062.5;	mu[17]=1250;

	//for s = 1:N
	//	B(s, :) = mvnpdf(observations', mu(:, s)', Sigma(:, :, s));
	//end
	float *B; // NxT
	B =	mvnpdf(observations,mu,Sigma,N,T,D); 


	//---------------------------------------------------------------//
	//						forward algorithm
	//---------------------------------------------------------------//
	double log_likelihood=0;
	double *alpha=(double*)calloc(N*T,sizeof(double)); // initialize to be zeros

	double *A_al=(double*)calloc(N,sizeof(double)); 
	double tmp;
	double alpha_sum;


	for(i=0;i<T;i++){
		if(i==0){
			for(j=0;j<N;j++){
				alpha[j*T+i] = B[j*T+i] * prior[j];
				// printf("%10e\n",alpha[j*T+i]);
			}

		}else{
			// alpha(:, t) = B(:, t) .* (A' * alpha(:, t - 1));
			for(j=0;j<N;j++){//col 
				tmp = 0;
				for(k=0;k<N;k++){ //row 
					tmp += A[k*N+j]*alpha[k*T+i-1];	
				}
				A_al[j] = tmp;
			}

			for(j=0;j<N;j++){
				alpha[j*T+i] = B[j*T+i] * A_al[j];
				//if(i==1){
				//	printf("%10e\n",alpha[j*T+i]);			
				//}
			}
		}	

		// alpha_sum      = sum(alpha(:, t));
		// alpha(:, t)    = alpha(:, t) ./ alpha_sum; 
		alpha_sum = 0;
		for(j=0;j<N;j++){
			alpha_sum += alpha[j*T+i];
		}
		for(j=0;j<N;j++){
			alpha[j*T+i] /= alpha_sum;
			//if(i==1 || i==0){
			//	printf("%10e\n",alpha[j*T+i]);			
			//}
		}

		log_likelihood += log(alpha_sum); 
	}
	printf("log_likelihood=%lf\n",log_likelihood);


	//---------------------------------------------------------------//
	//						backward algorithm
	//---------------------------------------------------------------//
	// beta:  NxT
	double *beta=(double*)calloc(N*T,sizeof(double)); // initialize to be zeros
	// beta(:, T) = ones(N, 1);
	for(i=0;i<N;i++){
		beta[i*T+T-1] = 1;	
	}
	//---------------------
	//for t = (T - 1):-1:1
	//    beta(:, t) = A * (beta(:, t + 1) .* B(:, t + 1));
	//    beta(:, t) = beta(:, t) ./ sum(beta(:, t));
	//end
	//---------------------
	double beta_sum;
	double *beta_B = (double*)malloc(sizeof(double)*N);
	for(i=T-2;i>=0;i--){ // cols backwards
		for(j=0;j<N;j++){
			beta_B[j] = beta[j*T+i+1]*B[j*T+i+1];	
		}	

		beta_sum = 0;
		for(j=0;j<N;j++){
			tmp = 0;
			for(k=0;k<N;k++){
				tmp += A[j*N+k] * beta_B[k];
			}
			beta[j*T+i] = tmp;
			beta_sum += tmp;
		}

		for(j=0;j<N;j++){
			beta[j*T+i] /= beta_sum;
		}
	}
	/*
	   for(i=0;i<N;i++){
	   printf("N=%d\n",i+1);
	   for(j=0;j<T;j++){
	   printf("%lf\n",beta[i*T+j]);
	   }	
	   printf("\n-----------------\n");
	   }
	 */




	//---------------------------------------------------------------//
	//						Expectation-Maximization 
	//---------------------------------------------------------------//

	double *xi_sum	= (double*)calloc(N*N,sizeof(double));
	double *gamma	= (double*)calloc(N*T,sizeof(double));

	double *alpha_beta_B = (double*)malloc(N*N*sizeof(double));
	double *xi_sum_tmp 	 = (double*)malloc(N*N*sizeof(double));
	double *gamma_norm	 = (double*)malloc(N*sizeof(double));


	//---------------------
	//for t = 1:(T - 1)
	//    xi_sum      = xi_sum + normalise(A .* (alpha(:, t) * (beta(:, t + 1) .* B(:, t + 1))'));
	//    gamma(:, t) = normalise(alpha(:, t) .* beta(:, t));
	//end
	//---------------------
	for(i=0;i<T-1;i++){
		for(j=0;j<N;j++){
			beta_B[j] = beta[j*T+i+1] * B[j*T+i+1];
		} // 1xN 
		// alpha: NxT
		for(j=0;j<N;j++){
			for(k=0;k<N;k++){
				alpha_beta_B[j*N+k]= alpha[j*T+i]*beta_B[k];			
			}
		}
		// A.*alpha_beta_B
		for(j=0;j<N;j++){
			for(k=0;k<N;k++){
				xi_sum_tmp[j*N+k]=A[j*N+k]*alpha_beta_B[j*N+k];
			}
		}
		// normalise
		normalise(xi_sum_tmp,N,N);
		// update xi_sum
		for(j=0;j<N;j++){
			for(k=0;k<N;k++){
				xi_sum[j*N+k] += xi_sum_tmp[j*N+k];
			}
		}

		for(j=0;j<N;j++){
			gamma_norm[j] = alpha[j*T+i] * beta[j*T+i];
		} // Nx1 
		normalise(gamma_norm,N,1);
		for(j=0;j<N;j++){
			gamma[j*T+i]= gamma_norm[j];
		} 
	}
	// gamma(:, T) = normalise(alpha(:, T) .* beta(:, T));
	for(j=0;j<N;j++){
		gamma_norm[j] = alpha[j*T+T-1] * beta[j*T+T-1];
	}
	normalise(gamma_norm,N,1);
	for(j=0;j<N;j++){
		gamma[j*T+T-1]= gamma_norm[j];
	} 

	/*
	   printf("xi_sum=\n");
	   for(j=0;j<N;j++){
	   for(k=0;k<N;k++){
	   printf("%10lf ",xi_sum[j*N+k]);	
	   }
	   printf("\n");
	   }

	   printf("gamma=\n");
	   for(i=0;i<N;i++){
	   printf("N=%d\n",i+1);
	   for(j=0;j<T;j++){
	   printf("%lf\n",gamma[i*T+j]);
	   }	
	   printf("\n-----------------\n");
	   }
	 */


	//------------------------------------------------------------// 
	//	output parameters
	//------------------------------------------------------------// 
	double *exp_prior=(double*)malloc(sizeof(double)*N); // Nx1
	for(i=0;i<N;i++){
		exp_prior[i] = gamma[i*T];
		//printf("%lf\n",exp_prior[i]);
	}

	double *exp_A=(double*)malloc(sizeof(double)*N*N); // NxN
	mk_stochastic(xi_sum,N,N,exp_A);
	/*
	   for(i=0;i<N;i++){
	   for(j=0;j<N;j++){
	   printf("%lf ",exp_A[i*N+j]);
	   }
	   printf("\n");
	   }	
	 */

	double *exp_mu			=(double*)calloc(N*D,sizeof(double));// DxN = 6x3
	double *exp_sigma		=(double*)calloc(N*D*D,sizeof(double));
	double *gamma_state_sum	=(double*)calloc(N,sizeof(double)); // Nx1

	//gamma_state_sum = sum(gamma, 2);
	for(i=0;i<N;i++){
		tmp = 0;
		for(j=0;j<T;j++){
			tmp += gamma[i*T+j];
		}
		//Set any zeroes to one before dividing to avoid NaN
		if(tmp==0){
			tmp = 1;
		}
		gamma_state_sum[i]=tmp;
		//printf("%lf\n",gamma_state_sum[i]);
	}	

	//---------------------//
	// matlab:
	// for s = 1:N
	//    gamma_observations = observations .* repmat(gamma(s, :), [D 1]);
	//    expected_mu(:, s)  = sum(gamma_observations, 2) / gamma_state_sum(s); 	%  updated mean
	//    expected_Sigma(:, :, s) = symmetrize(gamma_observations * observations' / gamma_state_sum(s) - ...
	//        expected_mu(:, s) * expected_mu(:, s)');	% updated covariance
	// end
	//---------------------//


	// gamma:NxT
	// observations:TxD (need to think it as DxT to avoid transposing) 
	// gamma_obs: DxT
	double *gamma_obs		=(double*)malloc(sizeof(double)*D*T);
	double *gamma_obs_mul	=(double*)malloc(sizeof(double)*D*D);
	double *exp_mu_mul		=(double*)malloc(sizeof(double)*D*D);
	double *obs_div_mu		=(double*)malloc(sizeof(double)*D*D);

	int offset;
	int start;
	int n;
	for(k=0;k<N;k++){
		//repmat gamma
		offset = k*T;
		for(i=0;i<D;i++){ // num of repeats
			start = i*T;
			for(j=0;j<T;j++){
				gamma_obs[start+j]=gamma[j+offset];	
			}	
		}
		// now gamma_obs is DxT
		// dot multiplication with observations
		for(i=0;i<D;i++){
			for(j=0;j<T;j++){
				gamma_obs[i*T+j] *= observations[j*D+i];	
			}
		}
		/*
		   if(k==0){
		   for(i=0;i<D;i++){
		   printf("gamma_obs,  round %d\n",i+1);
		   for(j=0;j<T;j++){
		   printf("%10lf\n", gamma_obs[i*T+j]);
		   }
		   printf("\n---------------------\n");
		   }
		   }
		 */

		//update exp_mu : DxN
		for(i=0;i<D;i++){
			tmp=0;
			for(j=0;j<T;j++){
				tmp += gamma_obs[i*T+j];	
			}
			exp_mu[i*N+k] = tmp/gamma_state_sum[k]; 
		}
		/*
		   if(k==0){
		   for(i=0;i<D;i++){
		   for(j=0;j<N;j++){
		   printf("%lf ",exp_mu[i*N+j]);	
		   }
		   printf("\n");
		   }
		   }
		 */

		// prepare symmetrize
		// nominator : gamma_obs(DxT)  *  observations (TxD)
		for(i=0;i<D;i++){ // row
			for(j=0;j<D;j++){ // col
				tmp = 0;	
				for(n=0;n<T;n++){
					tmp += gamma_obs[i*T+n]*observations[n*D+j];
				}
				gamma_obs_mul[i*D+j] = tmp;
			}
		}
		/*
		   if(k==0){
		   for(i=0;i<D;i++){
		   for(j=0;j<D;j++){
		   printf("%10e ", gamma_obs_mul[i*D+j]);
		   }
		   printf("\n");
		   }

		   }
		 */

		// denominator:
		for(i=0;i<D;i++){// row
			for(j=0;j<D;j++){ // col
				exp_mu_mul[i*D+j] = gamma_state_sum[k] - exp_mu[i*N+k]*exp_mu[j*N+k];
				if(k==0)	printf("%10e ",exp_mu_mul[i*D+j]);
			}	
			if(k==0)	printf("\n");
		}

		// matrix division:    obs_div_mu = gamma_obs_mul (DxD) /	exp_mu_mul (DxD)
		matrixDiv(gamma_obs_mul,D,D, exp_mu_mul,D,D, obs_div_mu);



		// write symmetrize()	
	}




	// may need to keep
	free(obs_div_mu);
	free(exp_mu_mul);
	free(gamma_obs_mul);
	free(gamma_obs);
	free(exp_sigma);
	free(exp_mu);
	free(exp_prior);
	free(exp_A);

	// release
	free(gamma_norm);
	free(xi_sum_tmp);
	free(alpha_beta_B);
	free(xi_sum);
	free(gamma);
	free(beta_B);
	free(beta);
	free(A_al);
	free(alpha);
	free(mu);
	free(A);
	free(prior);
	free(observations);





	// release
	free(prior);
	for(i=0;i<N;++i)
	{
		free(A[i]);
	}
	free(A);
	free(mu);
}


float *cov(float *A, unsigned int R, unsigned int C)
{
	unsigned int i,j;

	float *result;
	result = (float*) malloc(sizeof(float)*C*C);

	float *means;
	means = (float*) malloc(sizeof(float)*C);

	double sum;
	for(i=0;i<C;++i){
		sum = 0.0;
		for(j=0;j<R;++j){
			sum = sum + A[j*C+i]; // sum along each column	
		}
		means[i] = (float)(sum/(float)R);
	}

	// work on the lower triangular part of the matrix
	// then update the upper part

	// this can be optimized
	unsigned int m;
	for(i=0;i<C;i++){	// target row
		for(j=0;j<=i;j++){ // target column
			if(i==j){ // diagonal
				sum = 0.0;
				for(m=0;m<R;m++){
					sum =  sum + pow((A[m*C+j]-means[j]),2.0);
				}		
				result[i*C+j] = (float)(sum/(float)(R-1));
			}else{
				//  covariance between j column and i column	
				sum = 0.0;
				for(m=0;m<R;m++){
					sum =  sum + (A[m*C+i]-means[i])*(A[m*C+j]-means[j]); 	
				}		
				result[i*C+j] = (float)(sum/(float)(R-1));	
				result[j*C+i] = result[i*C+j];
			}
		}
	}

	return result;
}


float *repmat(float *input,unsigned int row, unsigned int col, unsigned int repNum)
{
	float *output=(float*) malloc(sizeof(float)*row*col*repNum);
	unsigned int stride = row*col;

	// perf hint: use calloc first and only duplicate along the diagnal
	int i,j,k;
	float data;
	for(i=0;i<row;i++){
		for(j=0;j<col;j++){
			data = input[i*col+j];
			for(k=0;k<repNum;k++){
				output[i*col+j+k*stride] = data;	
			}
		}
	}

	return output;
}


// 85x6, 1x6 , 6x6


// hint: obs is transposed
float *mvnpdf(float **obs, float *mean, float *cov, unsigned int N, unsigned int T, unsigned int D)
{
	// obs: TxD
	// mean : NxD
	// cov: DxDxN
	// we need to generate multi-variate probability density function for N hidden states

	unsigned int i,j,k,n;
	unsigned int stride = D*D;

	float *B=(float*)malloc(sizeof(float)*N*T);// NxT

	float *X0=(float*)malloc(sizeof(float)*T*D);// TxD
	float *mu = (float*)malloc(sizeof(float)*D); // 1xD
	float *sigma = (float*)malloc(sizeof(float)*D*D);  // DXD


	float *R;
	float *xRinv;
	xRinv = (float*)malloc(sizeof(float)*D*T); 
	float *tmpx;
	tmpx = (float*)malloc(sizeof(float)*D*T); 


	float tmp;
	float logSqrtDetSigma;
	float *quadform; // Tx1
	quadform=(float*)malloc(sizeof(float)*T);

	for(k=0;k<N;k++){ 
		//fetch mu	
		for(i=0;i<D;i++){
			mu[i] = mean[k*D+i];
		}
		//fetch sigma
		for(i=0;i<D;i++){
			for(j=0;j<D;j++){
				sigma[i*D+j] = cov[k*stride+i*D+j];
			}
		}

		// X0 = bsxfun(@minus,XY,mu);
		for(i=0;i<T;i++){
			for(j=0;j<D;j++){
				//X0[i*D+j]= obs[i*D+j]-mu[j];	
				X0[i*D+j]= obs[j][i]-mu[j];	
			}	
		}


		// R = chol(sigma);
		// this is the square matrix
		R = chol(sigma,D,D);
		/*
		   printf("R(%d)\n",k+1);
		   for(i=0;i<D;i++){
		   for(j=0;j<D;j++){
		   printf("%lf ",R[i*D+j]);
		   }
		   printf("\n");
		   }
		 */
		float *L;
		float *U;
		float *P;

		L = (float*)calloc(D*D,sizeof(float));
		U = (float*)calloc(D*D,sizeof(float));
		P = (float*)calloc(D*D,sizeof(float));

		LUdcmp(R,L,U,P,D);

		/*
		   printf("L=%d\n",k+1);
		   for(i=0;i<D;i++){
		   for(j=0;j<D;j++){
		   printf("%lf ",L[i*D+j]);
		   }
		   printf("\n");
		   }

		   printf("U=%d\n",k+1);
		   for(i=0;i<D;i++){
		   for(j=0;j<D;j++){
		   printf("%lf ",U[i*D+j]);
		   }
		   printf("\n");
		   }
		 */

		// xRinv*(L*U) =X0
		// backward substitution:  tmpx * L = X0
		for(j=0;j<T;j++){// each row 
			for(i=(D-1);i>=0;i=i-1){ // from last col to first col 
				if(i==(D-1)){
					tmpx[j*D+i] = X0[j*D+i]/L[i*D+i];
				}else{
					// reuse tmp
					tmp =0;
					for(n=D-1;n>i;n--){
						tmp = tmp + L[i*D+n]*tmpx[j*D+n];
					}
					tmpx[j*D+i] = (X0[j*D+i] - tmp)/L[i*D+i];	
				}
			}
		}
		/*
		   printf("step %d : tmpx=\n", k+1);
		   for(i=0;i<T;i++){
		   for(j=0;j<D;j++){
		   printf("%10lf ",tmpx[i*D+j]);
		   }
		   printf("\n");
		   }
		   printf("\n");
		 */

		// forward substitution
		// xRinv*U = tmpx
		for(j=0;j<T;j++){ // each row
			for(i=0;i<D;i++){// each column
				if(i==0){
					xRinv[j*D+i] = tmpx[j*D+i]/U[i*D+i];
				}else{
					// reuse tmp
					tmp =0;
					for(n=0;n<i;n++){
						tmp = tmp + U[i*D+n]*tmpx[j*D+n];
					}
					xRinv[j*D+i] = (tmpx[j*D+i] - tmp)/U[i*D+i];	
				}

			}
		}
		/*
		   printf("step %d : xRinv=\n", k+1);
		   for(i=0;i<T;i++){
		   for(j=0;j<D;j++){
		   printf("%10lf ",xRinv[i*D+j]);
		   }
		   printf("\n");
		   }
		   printf("\n");
		 */

		logSqrtDetSigma=0;
		for(i=0;i<D;i++){
			logSqrtDetSigma = logSqrtDetSigma + log(R[i*D+i]);	
		}	
		//printf("logSqrtDetSigma=%lf\n",logSqrtDetSigma);

		// quadform = sum(xRinv.^2, 2);
		//printf("\n--------------------------\n");
		//printf("quadform:\n");
		for(i=0;i<T;i++){
			tmp=0;
			for(j=0;j<D;j++){
				tmp += pow(xRinv[i*D+j],2);	
			}
			quadform[i]=tmp;
			//printf("%lf\n",quadform[i]);
		}

		//d=size(XY,2); // d=2
		//y = exp(-0.5*quadform - logSqrtDetSigma - d*log(2*pi)/2);
		tmp= logSqrtDetSigma + log(TWOPI)*D/2.0;

		//if(k==0) printf("\n");
		for(i=0;i<T;i++){
			B[i+k*T] = exp(-0.5*quadform[i]- tmp);	
			//if(k==0) printf("%10e\n",B[i]);
		}
		//if(k==0) printf("\n");



		free(L);
		free(U);
		free(P);

	}

	//release
	free(quadform);
	free(xRinv);
	free(tmpx);
	free(X0);
	free(mu);
	free(sigma);

	return B;
}

void LU_pivot(float *R, float *P, unsigned int n)
{
	unsigned int i,j,k;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			//if(i==j)	P[i*n+j]=1;	
			P[i*n+j]=(i==j);
		}
	}

	unsigned int pivot_row;
	float tmp;
	// find the pivot value for each column
	for(i=0;i<n;i++){ // each column
		pivot_row = i;	
		for(j=i;j<n;j++){ // go through rows
			if(fabs(R[j*n+i]) > fabs(R[pivot_row*n+i])) pivot_row = j;	
		}
		// interchange the two rows
		if(pivot_row != i){
			for(k=0;k<n;k++){
				tmp = P[i*n+k];
				P[i*n+k] = P[pivot_row*n+k];
				P[pivot_row*n+k] = tmp;
			}	
		}

	}

}


void LUdcmp(float *sigma, float *LowerTri, float *UpperTri, float *permM, unsigned int n)
{
	unsigned int i,j,k;
	float tmp;

	float *input,*P,*L,*U;
	input = sigma;
	L = LowerTri;
	U = UpperTri;
	P = permM;

	LU_pivot(input,P,n);
	/*	
		printf("P=\n");
		for(i=0;i<n;i++){
		for(j=0;j<n;j++){
		printf("%d ",(int)P[i*n+j]);
		}
		printf("\n");
		}
	 */


	float *primeMat;
	primeMat = (float *)calloc(n*n,sizeof(float));

	// P*input
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			tmp = 0;
			for(k=0;k<n;k++){
				tmp = tmp + P[i*n+k]*input[k*n+j];
			}   
			primeMat[i*n+j] = tmp;
		}
	}
	/*	
		printf("primeMat=\n");
		for(i=0;i<n;i++){
		for(j=0;j<n;j++){
		printf("%lf ",primeMat[i*n+j]);
		}
		printf("\n");
		}
	 */

	// set the diagnal of L to be ones
	for(i=0;i<n;i++){
		L[i*n+i] = 1;
	}

	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			// Upper triangular matrix
			if(j<=i){
				tmp = 0;
				for(k=0;k<j;k++){
					tmp = tmp + L[j*n+k]*U[k*n+i];  
				}
				U[j*n+i] = primeMat[j*n+i] - tmp;   
			} 

			// Lower triangular matrix
			if(j>=i){
				tmp = 0;
				for(k=0;k<i;k++){
					tmp = tmp + L[j*n+k]*U[k*n+i];  
				}
				L[j*n+i] = (primeMat[j*n+i] - tmp)/U[i*n+i];    
			}

		}
	}

	// release
	free(primeMat);
}

