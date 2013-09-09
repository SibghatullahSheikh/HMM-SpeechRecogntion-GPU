#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "train.h"
//#include "feature_extraction.h"

#define PI  M_PI    /* pi to machine precision, defined in math.h */
#define TWOPI   (2.0*PI)
#define MAX(a,b) ((a) > (b) ? a : b)

void Train(float **observations, unsigned int featureDim, unsigned int frameNum, unsigned int hiddenstates)
{
	// observations[D][T]

	unsigned int i,j,k;
	unsigned int D = featureDim;
	unsigned int T = frameNum; 
	unsigned int N = hiddenstates; // hidden states

	// %prior = normalise(rand(N, 1));
	float *prior = (float*) malloc(sizeof(float)*N);
	prior[0] = 0.555826367548231;
	prior[1] = 0.384816327750085;
	prior[2] = 0.0593573047016839;

//	for(i=0;i<N;++i)
//	{
//		printf("prior[%d]=%f \t", i, prior[i]);
//	}
//	printf("\n\n");
//

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

//	for(i=0;i<N;++i)
//	{
//		for(j=0;j<N;++j)
//		{
//			printf("A[%d][%d]=%f\t",i,j,A[i][j]);	
//		}
//		printf("\n");
//	}
//

	//---------------------------------------------------------
	// Sigma = repmat(diag(diag(cov(observations'))), [1 1 N]);
	//---------------------------------------------------------

	// find transposition of observation(dxt)
	float **observations_t= (float**)malloc(sizeof(float*)*T);
	for(i=0;i<T;++i)
	{
		observations_t[i] = (float*)malloc(sizeof(float)*D);
	}
	
	transpose(observations, observations_t, D, T);

//	for(i=0;i<T;++i){
//		for(j=0;j<D;++j){
//			printf("%0.7f\t",observations_t[i][j]);
//		}
//		printf("\n");
//	}		   
//


	float **diag_diag_cov = (float**)malloc(sizeof(float*)*D);
	for(i=0;i<D;++i)
	{
		diag_diag_cov[i] = (float*)malloc(sizeof(float)*D);
	}

	//init_2d_f(diag_diag_cov,D,D,0.f);

	// in 	TxD 
    // out	DxD
	cov(observations_t,T,D,diag_diag_cov);

	// clear the off-diagonal elements
	for(i=0;i<D;i++){
		for(j=0;j<D;j++){
			if(i!=j){
				diag_diag_cov[i][j]=0.f;	
			}
		}
	}		   

//	for(i=0;i<D;++i){
//		for(j=0;j<D;++j){
//			printf("%0.4e ",diag_diag_cov[i][j]);
//		}
//		printf("\n");
//	}		   


	// Sigma = repmat(diag_diag_cov,D,D,N); // repeat (DxD) N times
	float ***Sigma;
	Sigma = (float***)malloc(sizeof(float**)*D); // row
	for(i=0;i<D;++i){
		Sigma[i] = (float**) malloc(sizeof(float*)*D); // col
		for(j=0;j<D;++j){
			Sigma[i][j] = (float*) malloc(sizeof(float)*N);
			for(k=0;k<N;++k){
				Sigma[i][j][k] = 0.f;	
			}
		}	
	}

	for(i=0;i<D;++i){
		for(j=0;j<D;++j){
			for(k=0;k<N;++k){
				Sigma[i][j][k] = diag_diag_cov[i][j];	
			}
		}	
	}

//	for(k=0;k<N;++k){
//		printf("N=%d\n",k+1);
//		for(i=0;i<D;++i){
//			for(j=0;j<D;++j){
//				printf("%.5e ",Sigma[i][j][k]);	
//			}
//			printf("\n");
//		}	
//		printf("\n");
//	}
//

	// mu = observations(:, ind(1:N));   % initialize the mean value
	// DxN
	float **mu= (float**)malloc(sizeof(float*)*D);
	for(i=0;i<D;++i){
		mu[i] = (float*)malloc(sizeof(float)*N);
	}
	
	mu[0][0]=875;		mu[0][1]=1000;		mu[0][2]=687.5;						
	mu[1][0]=562.5;		mu[1][1]=1562.5;	mu[1][2]=875;	
	mu[2][0]=187.5;		mu[2][1]=750;		mu[2][2]=562.5;		
	mu[3][0]=1375; 		mu[3][1]=875;		mu[3][2]=1562.5;	
	mu[4][0]=1187.5;	mu[4][1]=1437.5;	mu[4][2]=1062.5; 	
	mu[5][0]=1625;		mu[5][1]=1187.5;	mu[5][2]=1250;
	
	
	float **mu_t= (float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		mu_t[i] = (float*)malloc(sizeof(float)*D);
	}
	
	transpose(mu, mu_t, D, N);


	//for s = 1:N
	//	B(s, :) = mvnpdf(observations', mu(:, s)', Sigma(:, :, s));
	//end

	// NxT
	// use double to tolerate the extreme small value
	double **B = (double**)malloc(sizeof(double*)*N);
	for(i=0;i<N;++i){
		B[i] = (double*)malloc(sizeof(double)*T);
	}

	mvnpdf(B,observations_t, mu_t,Sigma,N,T,D); 

/*
	for(i=0;i<N;++i){
		printf("N=%d", i+1);
		for(j=0;j<T;++j){
			printf("%.5e\n",B[i][j]);
		}
	}
*/

/*
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
//	
//	   for(i=0;i<N;i++){
//	   printf("N=%d\n",i+1);
//	   for(j=0;j<T;j++){
//	   printf("%lf\n",beta[i*T+j]);
//	   }	
//	   printf("\n-----------------\n");
//	   }
//	 




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

//
//	   printf("xi_sum=\n");
//	   for(j=0;j<N;j++){
//	   for(k=0;k<N;k++){
//	   printf("%10lf ",xi_sum[j*N+k]);	
//	   }
//	   printf("\n");
//	   }
//
//	   printf("gamma=\n");
//	   for(i=0;i<N;i++){
//	   printf("N=%d\n",i+1);
//	   for(j=0;j<T;j++){
//	   printf("%lf\n",gamma[i*T+j]);
//	   }	
//	   printf("\n-----------------\n");
//	   }
//


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
//	
//	   for(i=0;i<N;i++){
//	   for(j=0;j<N;j++){
//	   printf("%lf ",exp_A[i*N+j]);
//	   }
//	   printf("\n");
//	   }	
//

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
	//	
	//	   if(k==0){
	//	   for(i=0;i<D;i++){
	//	   printf("gamma_obs,  round %d\n",i+1);
	//	   for(j=0;j<T;j++){
	//	   printf("%10lf\n", gamma_obs[i*T+j]);
	//	   }
	//	   printf("\n---------------------\n");
	//	   }
	//	   }
	//	 

		//update exp_mu : DxN
		for(i=0;i<D;i++){
			tmp=0;
			for(j=0;j<T;j++){
				tmp += gamma_obs[i*T+j];	
			}
			exp_mu[i*N+k] = tmp/gamma_state_sum[k]; 
		}
	
//		   if(k==0){
//		   for(i=0;i<D;i++){
//		   for(j=0;j<N;j++){
//		   printf("%lf ",exp_mu[i*N+j]);	
//		   }
//		   printf("\n");
//		   }
//		   }
//

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
//		
//		   if(k==0){
//		   for(i=0;i<D;i++){
//		   for(j=0;j<D;j++){
//		   printf("%10e ", gamma_obs_mul[i*D+j]);
//		   }
//		   printf("\n");
//		   }
//
//		   }
//		 

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
*/




	// release
	free(prior);
	for(i=0;i<N;++i){
		free(A[i]);
		free(mu_t[i]);
		free(B[i]);
	}
	for(i=0;i<T;++i){
		free(observations_t[i]);
	}
	for(i=0;i<D;++i){
		free(diag_diag_cov[i]);
		free(mu[i]);
	}
	// 3D sigma
	for(i=0;i<D;++i){
		for(j=0;j<D;++j){
			free(Sigma[i][j]);
		}
	}
	for(i=0;i<D;++i){
		free(Sigma[i]);
	}


	free(A);
	free(observations_t);
	free(diag_diag_cov);
	free(Sigma);
	free(mu);
	free(B);

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



void cov(float **input, unsigned int R, unsigned int C, float **result)
{
	unsigned int i,j,m;

	float *means;
	means = (float*) malloc(sizeof(float)*C);

	double sum;
	for(i=0;i<C;++i){
		sum = 0.0;
		for(j=0;j<R;++j){
			sum = sum + input[j][i]; // sum along each column	
		}
		means[i] = (float)(sum/(float)R);
	}

	// work on the lower triangular part of the matrix
	// then update the upper part

	// this can be optimized
	for(i=0;i<C;++i){	
		for(j=0;j<=i;++j){ 
			//  covariance between j column and i column	
			sum = 0.0;
			for(m=0;m<R;++m){
				sum =  sum + (input[m][i]-means[i])*(input[m][j]-means[j]); 	
			}		
			result[j][i] = (float)(sum/(float)(R-1));	
		}
	}

}

// 85x6, 1x6 , 6x6
void mvnpdf(double **B, float **obs, float **mean, float ***cov, unsigned int N, unsigned int T, unsigned int D)
{
	// out:	NxT
	// obs: TxD
	// mean : NxD
	// cov: DxDxN
	// we need to generate multi-variate probability density function for N hidden states

	unsigned int i,j,k,n;
	//unsigned int stride = D*D;

	double tmp;
	float data;

	float logSqrtDetSigma;


	float **X0=(float**)malloc(sizeof(float*)*T);// TxD
	for(i=0;i<T;++i){
		X0[i] = (float*)malloc(sizeof(float)*D);
	}

	float *mu = (float*)malloc(sizeof(float)*D); // 1xD

	float **sigma  = 	(float**)malloc(sizeof(float*)*D); // DxD
	float **R 	   =	(float**)malloc(sizeof(float*)*D); // DxD
	float **R_pinv =	(float**)malloc(sizeof(float*)*D); // DxD
	for(i=0;i<D;++i){
		sigma[i] 	= (float*)malloc(sizeof(float)*D);
		R[i] 	 	= (float*)malloc(sizeof(float)*D);
		R_pinv[i] 	= (float*)malloc(sizeof(float)*D);
	}

	float **xRinv = (float**)malloc(sizeof(float*)*T);// TxD
	float **tmpx = (float**)malloc(sizeof(float*)*T);// TxD
	for(i=0;i<T;++i){
		xRinv[i] = (float*)malloc(sizeof(float)*D);
		tmpx[i] = (float*)malloc(sizeof(float)*D);
	}

	float *quadform; // Tx1
	quadform=(float*)malloc(sizeof(float)*T);


	
	for(k=0;k<N;++k){ 	// selected states
		//fetch mu	
		
		for(i=0;i<D;i++){
			mu[i] = mean[k][i];
		}
		//fetch sigma
		for(i=0;i<D;i++){
			for(j=0;j<D;j++){
				sigma[i][j] = cov[i][j][k];
			}
		}

		
		// X0 = bsxfun(@minus,XY,mu);
		for(i=0;i<T;++i){
			for(j=0;j<D;++j){
				//X0[i*D+j]= obs[i*D+j]-mu[j];	
				X0[i][j]= obs[i][j]-mu[j];	
			}	
		}


		// R = chol(sigma);
		// this is the square matrix
		chol(sigma,D,D,R);
		
		if(1){

			printf("R(%d)\n",k+1);
			for(i=0;i<D;++i){
				for(j=0;j<D;++j){
					printf("%f ",R[i][j]);
				}
				printf("\n");
			}
		}

		// xRinv*R =X0
		pinv_main(R,R_pinv,D);


		// find xRinv = x0 * R_pinv
		// implement matrix multiplication
		for(i=0;i<T;++i){ // each row
			for(j=0;j<D;++j){// each column
				tmp = 0.0;
				for(n=0;n<D;++n){
					tmp += X0[i][n]*R_pinv[n][j];
				}
				xRinv[i][j] = (float)tmp;	
			}
		}

		if(k==0 && 0){
			for(i=0;i<T;++i){
				for(j=0;j<D;++j){
					printf("%10.5f ",xRinv[i][j]);
				}
				printf("\n");
			}
		}

		logSqrtDetSigma=0.f;
		for(i=0;i<D;++i){
			logSqrtDetSigma = logSqrtDetSigma + log(R[i][i]);	
		}	
		//printf("logSqrtDetSigma=%.5f\n",logSqrtDetSigma);

		// quadform = sum(xRinv.^2, 2);
		for(i=0;i<T;++i){
			tmp=0.0;
			for(j=0;j<D;++j){
				tmp += pow(xRinv[i][j],2.0);	
			}
			quadform[i]= (float)tmp;
			//printf("%.5f\n",quadform[i]);
		}

		//d=size(XY,2); // d=2
		//y = exp(-0.5*quadform - logSqrtDetSigma - d*log(2*pi)/2);
		data = logSqrtDetSigma + log(TWOPI)*(float)D/2.0;
		//printf("%lf\n",tmp);

		// output
		
		printf("k=%d\n",k);
		for(i=0;i<T;++i){
			//B[k][i] = exp(-0.5*quadform[i] - data); 
			tmp = exp(-0.5*quadform[i] - data); 
			//printf("%e\n",tmp);
			B[k][i] = tmp;
		}



	}



	//release
	for(i=0;i<T;++i)
	{
		free(X0[i]);
		free(xRinv[i]);
		free(tmpx[i]);
	}

	for(i=0;i<D;++i)
	{
		free(sigma[i]);
		free(R[i]);
		free(R_pinv[i]);
	}

	free(sigma);
	free(R);
	free(R_pinv);
	free(xRinv);
	free(tmpx);
	free(X0);
	free(mu);

	free(quadform);

}


void chol(float **sigma, unsigned int row, unsigned int col, float **C)
{
    // cholesky fractorization
    // reference : http://web.cecs.pdx.edu/~gerry/nmm/mfiles/linalg/Cholesky.m
    if(row != col){
        printf("Sigma must be square. file:%s,line:%d\n", __FILE__, __LINE__);
        exit(1);
    }

    unsigned int i,j,m;
    float s;
    double tmp;
    for(i=0;i<col;++i){
        for(j=i;j<col;++j){
            if(j==0){
                s = sigma[i][i]; // i=0,j=0 is special case
            }else{
                tmp=0.0;
                for(m=0;m<=i;++m){
                    tmp = tmp + C[m][i]*C[m][j];
                }
                s = sigma[i][j] - (float)tmp;
                //s = tmp;  
                // printf("%lf\n",tmp);
            }

            if(j>i){
                C[i][j] = s/C[i][i];
            }else{
                //if(s<=0){
                //    printf("C is not positive definite to working precision. file:%s, line:%d\n", __FILE__, __LINE__);
                //    exit(1);
                //}
                C[i][i] = sqrt(s);
            }
            //printf("%lf \n", C[i*col+i]);
        }
        //printf("\n");
    }
}

void pinv_main(float **R, float **R_pinv, unsigned int D)
{
	// input : R
	// output: R_pinv
	// D: row/col

	// use singular value decomposition to find pseudo inverse of a matrix

	unsigned int i,j;
	unsigned int n = D;
	float **input;
	input = R;

	// copy input to U
	float **U;
	U = (float**)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		U[i] = (float*)malloc(sizeof(float)*n);
	}

	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			U[i][j] = input[i][j];
		}
	}

	// identity matrix V
	//eye(5);
	float **V;
	V = (float **)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		V[i] = (float*)malloc(sizeof(float)*n);
	}

	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			V[i][j] = (i==j)? 1.f: 0.f;
		}
	}

	// S
	float **S;
	S = (float **)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		S[i] = (float*)malloc(sizeof(float)*n);
	}

	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			S[i][j] = 0.f;
		}
	}

	// svd
	svd(input,U,S,V,n);

/*
	printf("\n--------------------\nSVD Results:\n");
	printf("U=\n");
	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			printf("%10.5f \t", U[i][j]);
		}
		printf("\n");
	}

	printf("V=\n");
	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			printf("%10.5f \t", V[i][j]);
		}
		printf("\n");
	}

	printf("S=\n");
	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			printf("%10.5f \t", S[i][j]);
		}
		printf("\n");
	}
*/

	// pseudo inverse
	pinv(U,S,V,n,R_pinv);



	// release
	for(i=0;i<n;++i){
		free(U[i]);
		free(V[i]);
		free(S[i]);
	}
	free(U);
	free(V);
	free(S);

}


void svd(float **input, float **U, float **S, float **V, unsigned int n)
{
	// parameters
	unsigned int MAX_STEPS = 100;
	double TOL = 1e-8;
	unsigned int steps,j,k,ind;
	double converge;
	double alpha,beta,gamma,zeta;
	double t,c,s;
	float tmp;

	// debug
	int ii,jj;

	float *singvals= (float *)malloc(sizeof(float)*n); 

	for(steps=0; steps<=MAX_STEPS; ++steps)
	{
		converge = 0.0;	

		for(j=1;j<n;++j){
			for(k=0;k<=j-1;++k){
				// compute [alpha gamma; gamma beta] = (k,j) submatrix of U'*U
				
				//alpha=sum(U(:,k).*U(:,k));
				alpha = 0.0;
				for(ind=0;ind<n;++ind){
					alpha += U[ind][k]*U[ind][k];	
				}

				//beta=sum(U(:,j).*U(:,j));
				beta = 0.0;
				for(ind=0;ind<n;++ind){
					beta += U[ind][j]*U[ind][j];	
				}

      			//gamma=sum(U(:,k).*U(:,j));
				gamma = 0.0;
				for(ind=0;ind<n;++ind){
					gamma += U[ind][k]*U[ind][j];	
				}

				if(steps==0 && j==1 && k==0 && 0){
					printf("alpha=%lf \t beta=%lf	\t gamma=%lf \n", alpha, beta, gamma);
				}
			
				//converge=max(converge,abs(gamma)/sqrt(alpha*beta));
				converge = MAX(converge, fabs(gamma)/sqrt(alpha*beta));

				if(steps==0 && j==1 && k==0 && 0){
					printf("converge=%f \n", converge);
				}

				if(gamma != 0){
					zeta = (beta-alpha)/(gamma+gamma);	
					double s = (double)sign_d(zeta);
					t = s /(fabs(zeta) + sqrt(1+zeta*zeta));

					if(steps==0 && j==1 && k==0 && 0){
						printf("zeta=%lf \t t=%lf\n", zeta, t);
					}

				}else{
					t = 0.0;
				}

				c = 1/sqrt(1+t*t);
				s = c * t;

				if(steps==0 && j==1 && k==0 && 0){
					printf("c=%lf \t s=%lf\n", c, s);
				}


     			//update columns k and j of U
      			//T=U(:,k);
      			//U(:,k)=c*T-s*U(:,j);
      			//U(:,j)=s*T+c*U(:,j);
				for(ind=0;ind<n;++ind){
					tmp = U[ind][k];
					U[ind][k] =  (float)c*tmp - (float)s*U[ind][j];
					U[ind][j] =  (float)s*tmp + (float)c*U[ind][j];
				}

				if(steps==0 && j==1 && k==0 && 0){
					printf("U=\n");
					for(ii=0;ii<n;++ii){
						for(jj=0;jj<n;++jj){
							printf("%lf \t", U[ii][jj]);
						}
						printf("\n");
					}
				}
				
      
				//update matrix V of right singular vectors
				//T=V(:,k);    
				//V(:,k)=c*T-s*V(:,j);
				//V(:,j)=s*T+c*V(:,j);
				for(ind=0;ind<n;++ind){
					tmp = V[ind][k];
					V[ind][k] =  (float)c*tmp - (float)s*V[ind][j];
					V[ind][j] =  (float)s*tmp + (float)c*V[ind][j];
				}

				if(steps==0 && j==1 && k==0 && 0){

					printf("V=\n");
					for(ii=0;ii<n;++ii){
						for(jj=0;jj<n;++jj){
							printf("%lf \t", V[ii][jj]);
						}
						printf("\n");
					}
				}
				
			
			}	
		}

		if(converge < TOL){
			break;	
		}
	}

	if(steps == MAX_STEPS){
		printf("Jacobi-SVD fails to converge! file:%s, line:%d\n",__FILE__,__LINE__);
		exit(1);
	}


	//for j=1:n
	//	singvals(j)=norm(U(:,j));
	//	U(:,j)=U(:,j)/singvals(j);
	//end
	for(ind=0;ind<n;++ind){
		singvals[ind] = tmp = norm_square(U,ind,n); 
		for(j=0;j<n;++j){
			U[j][ind] = U[j][ind]/tmp;
		}
	}


	// S=diag(singvals);
	for(j=0;j<n;++j){
		for(k=0;k<n;++k){
			S[j][k] = (j==k)? singvals[j]: 0.f;
		}
	}

}


int sign_d(double x)
{
	int out;

	if(x>0){
		out = 1;
	}else if(x<0){
		out = -1;
	}else{
		out = 0;
	}
	
	return out;
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



void pinv(float **U, float **S, float **V, unsigned int n, float **X)
{
	unsigned int i,j,k;
	double tmp;

	// square matrix
	// s = diag(S);		double s=0.0;
	float **s;
	s = (float **)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		s[i] = (float *)malloc(sizeof(float)*n);
	}
	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			s[i][j] = (i==j)? S[i][i]:0.0;
		}
	}

	// for orthogonal matrix, the transposition and inversion is identical
	float **invU;
	invU = (float **)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		invU[i] = (float *)malloc(sizeof(float)*n);
	}
	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			invU[j][i] = U[i][j];
		}
	}

	// matT: temporary matrix
	float **matT;
	matT = (float **)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		matT[i] = (float *)malloc(sizeof(float)*n);
	}


	// tol = max(m,n) * eps(max(s));
	// choose a relative good tolerance value
	double tol =  1e-8;

	// r = sum(s > tol);
	// the number of value that are larger than the tol
	unsigned int r=0;
	for(i=0;i<n;++i){
		if(s[i][i] > tol){
			r += 1;
		}
	}

	if (r==0){
		//	X = zeros(size(A'),class(A));
		printf("Warning: the pinv() output will be zeros.\n file:%s, line:%d",__FILE__,__LINE__);
		for(i=0;i<n;++i){
			for(j=0;j<n;++j){
				X[i][j] = 0.0;	
			}
		}
	}else{
		// the first r values get reciprocal
		// s = diag(ones(r,1)./s(1:r));
		for(i=0;i<r;++i){
			s[i][i]	= 1.f/s[i][i];
			
		}

		// X = V(:,1:r)*s*U(:,1:r)';

		// take the first r cols of V, multiply with s (rxr)
		for(i=0;i<n;++i){ // rows
			for(j=0;j<r;++j){ // cols
				tmp = 0.0;
				for(k=0;k<r;++k){
					tmp += V[i][k]*s[k][j];	
				}	
				matT[i][j] = (float)tmp;
			}
		}

		// matT (nxr) * invU(rxn)
		for(i=0;i<n;++i){ // rows
			for(j=0;j<n;++j){ // cols
				tmp = 0.0;
				for(k=0;k<r;++k){
					tmp += matT[i][k]*invU[k][j];	
				}	
				X[i][j] = (float)tmp;
			}
		}
	}



	// release
	for(i=0;i<n;++i){
		free(s[i]);
		free(invU[i]);
		free(matT[i]);
	}
	free(s);
	free(invU);
	free(matT);
}


