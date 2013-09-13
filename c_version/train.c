#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "train.h"
#include "utils.h"

#define PI  M_PI    /* pi to machine precision, defined in math.h */
#define TWOPI   (2.0*PI)
#define MAX(a,b) ((a) > (b) ? a : b)
#define PP() printf("FILE: %s, LINE: %d\n",__FILE__,__LINE__) // print position


void TRAIN(float **observations, unsigned int featureDim, unsigned int frameNum, unsigned int hiddenstates)
{
	// observations[D][T]

	unsigned int i,j,k;
	unsigned int D = featureDim;
	unsigned int T = frameNum; 
	unsigned int N = hiddenstates; // hidden states

	// %prior = normalise(rand(N, 1));
	float *prior = (float*) malloc(sizeof(float)*N);

	//%A = mk_stochastic(rand(N));
	float **A = (float**) malloc(sizeof(float*)*N);
	for(i=0;i<N;++i)
	{
		A[i] = (float*) malloc(sizeof(float)*N);
	}

	float **diag_diag_cov = (float**)malloc(sizeof(float*)*D);
	for(i=0;i<D;++i)
	{
		diag_diag_cov[i] = (float*)malloc(sizeof(float)*D);
	}

	//init_2d_f(diag_diag_cov,D,D,0.f);

	// Sigma = repmat(diag(diag(cov(observations'))), [1 1 N]);
	float **observations_t= (float**)malloc(sizeof(float*)*T);
	for(i=0;i<T;++i){
		observations_t[i] = (float*)malloc(sizeof(float)*D);
	}

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

	float **mu= (float**)malloc(sizeof(float*)*D);
	for(i=0;i<D;++i){
		mu[i] = (float*)malloc(sizeof(float)*N);
	}

	unsigned int loopID;

	for(loopID=0;loopID<15;++loopID)
	{
		printf("loop = %d\n", loopID);

		if(loopID==0) // initialize parameters
		{
			prior[0] = 0.555826367548231;
			prior[1] = 0.384816327750085;
			prior[2] = 0.0593573047016839;
			
			A[0][0]=0.126558246942481;
			A[0][1]=0.438475341086527;
			A[0][2]=0.434966411970992;
			A[1][0]=0.459614415425850;
			A[1][1]=0.132462410695470;
			A[1][2]=0.407923173878680;
			A[2][0]=0.350943345583750;
			A[2][1]=0.355739578481455;
			A[2][2]=0.293317075934795;

			transpose(observations, observations_t, D, T);
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

			for(i=0;i<D;++i){
				for(j=0;j<D;++j){
					for(k=0;k<N;++k){
						Sigma[i][j][k] = diag_diag_cov[i][j];	
					}
				}	
			}

			mu[0][0]=875;		mu[0][1]=1000;		mu[0][2]=687.5;						
			mu[1][0]=562.5;		mu[1][1]=1562.5;	mu[1][2]=875;	
			mu[2][0]=187.5;		mu[2][1]=750;		mu[2][2]=562.5;		
			mu[3][0]=1375; 		mu[3][1]=875;		mu[3][2]=1562.5;	
			mu[4][0]=1187.5;	mu[4][1]=1437.5;	mu[4][2]=1062.5; 	
			mu[5][0]=1625;		mu[5][1]=1187.5;	mu[5][2]=1250;

			//  word.name="apple01";
			//	word.prior = prior;			
			//	word.A = A;
			//	word.sigma = Sigma;
			//	word.mu = mu;
			//	word.N = hiddenstates; 
			//	word.D = featureDim; 
			//	word.T = frameNum; 
		
		}

		// pass reference	
		train_sample(observations, observations_t, prior,A,Sigma,mu,N,D,T,loopID);

		if(loopID==0 && 0){
			
			check_2d_f(mu,D,N);	
			//check_1d_f(prior,N);	
			//check_2d_f(A,N,N);	
			//check_3d_f(Sigma,D,D,N);	
			//exit(1);
		}

	}	

	// save HMM()





	// release
	free(prior);
	for(i=0;i<N;++i){
		free(A[i]);
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
}


void train_sample(float **observations, float **observations_t,float*prior, float **A, float ***Sigma,float**mu, unsigned int N, unsigned int D, unsigned int T, unsigned int loopID)
{

	unsigned int i;

	float **mu_t= (float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		mu_t[i] = (float*)malloc(sizeof(float)*D);
	}
	
	transpose(mu, mu_t, D, N);

	if(loopID == 1 && 0){
		check_2d_f(mu,D,N);	

		puts("transpose");
		check_2d_f(mu_t,N,D);	
		exit(1);
	}



	//for s = 1:N
	//	B(s, :) = mvnpdf(observations', mu(:, s)', Sigma(:, :, s));
	//end

	// NxT
	// use double to tolerate the extreme small value
	float **B = (float **)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		B[i] = (float*)malloc(sizeof(float)*T);
	}

	if(loopID == 1 && 0){
		check_2d_f(observations_t,T,D);	
		PP();
		exit(1);
	}

	mvnpdf(B,observations_t, mu_t,Sigma,N,T,D,loopID); 

	if(loopID == 1 && 0){
		check_2d_f(B,N,T);	
		exit(1);
	}

//	for(i=0;i<N;++i){
//		printf("N=%d \n", i+1);
//		for(j=0;j<T;++j){
//			printf("%.5e\n",B[i][j]);
//		}
//	}



	//---------------------------------------------------------------//
	//						forward algorithm
	//---------------------------------------------------------------//

	float log_likelihood=0.0;
	// NxT
	float **alpha= (float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		alpha[i] = (float*)malloc(sizeof(float)*T);
	}

	Forward(B,A,prior,alpha,N,T,&log_likelihood);

	printf("log_likelihood=%f\n",log_likelihood);



	

	//---------------------------------------------------------------//
	//						backward algorithm
	//---------------------------------------------------------------//
	// NxT
	float **beta = (float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		beta[i] = (float*)malloc(sizeof(float)*T);
	}

	Backward(A,B,beta,N,T);
    
	//check_2d_f(beta,N,T);

	//---------------------------------------------------------------//
	//						Expectation-Maximization 
	//---------------------------------------------------------------//
	EM(observations,prior,A,mu,B,Sigma,alpha,beta,D,N,T);



	// release
	for(i=0;i<N;++i){
		free(mu_t[i]);
		free(B[i]);
		free(alpha[i]);
		free(beta[i]);
	}
	free(mu_t);
	free(B);
	free(alpha);
	free(beta);
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
void mvnpdf(float **B, float **obs, float **mean, float ***cov, unsigned int N, unsigned int T, unsigned int D, unsigned int loopID)
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

		if(k==0 && 0){

			printf("sigma(%d)\n",k+1);
			for(i=0;i<D;++i){
				for(j=0;j<D;++j){
					printf("%f ",sigma[i][j]);
				}
				printf("\n");
			}
		}


		// R = chol(sigma);
		// this is the square matrix
		cholcov(sigma,D,D,R);
		
		if(k==0 && 0){

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
		
		// printf("k=%d\n",k);
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

// Cholesky-like covariance decomposition
void cholcov(float **sigma, unsigned int row, unsigned int col, float **C)
{

	if(row != col){
		printf("Current cholcov() only works for square matrix! FILE:%s, LINE:%d",__FILE__,__LINE__);
		exit(1);
	}

	unsigned int n = row;
	unsigned int i,j,k;
	double tmp;

	//x = sigma+sigma';
	float **x = (float**) malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		x[i] = (float*) malloc(sizeof(float)*n);
	}

	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			x[i][j]	 = (sigma[i][j] + sigma[j][i])/2;
		}	
	}

//	printf("\nx=\n");
//	for(i=0;i<n;++i){
//		for(j=0;j<n;++j){
//			printf("%f ",x[i][j]);
//		}	
//		printf("\n");
//	}

//---------------------------
//	[u,s,v]=jacobi_svd(x/2);
//	U = v;
//	D = s;
//---------------------------
	float **U;
	U = (float**)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		U[i] = (float*)malloc(sizeof(float)*n);
	}

	float **u;
	u = (float **)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		u[i] = (float*)malloc(sizeof(float)*n);
	}

	float **D;
	D = (float **)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		D[i] = (float*)malloc(sizeof(float)*n);
	}

	svd(x,u,D,U,n);

//	printf("D=\n");
//	for(i=0;i<n;++i){
//		for(j=0;j<n;++j){
//			printf("%10.5e \t", D[i][j]);
//		}
//		printf("\n");
//	}
//
//	printf("U=\n");
//	for(i=0;i<n;++i){
//		for(j=0;j<n;++j){
//			printf("%10.5f \t", U[i][j]);
//		}
//		printf("\n");
//	}
//


//-------------------------------------------
//	[ignore,maxind] = max(abs(U),[],1);
//	negloc = (U(maxind + (0:n:(m-1)*n)) < 0);
//	U(:,negloc) = -U(:,negloc);
//-------------------------------------------


//-------------------------------------------
//	D = diag(D);
//	tol = eps(max(D)) * length(D);
//	t = (abs(D) > tol);
//	D = D(t);
//	p = sum(D<0); % number of negative eigenvalues
//	T = diag(sqrt(D)) * U(:,t)'
//-------------------------------------------
	float *diagD = 	(float*)malloc(sizeof(float)*n);
	int *t = (int*)malloc(sizeof(int)*n);

	float TOL = 1e-8;

	for(i=0;i<n;++i){
		diagD[i] = D[i][i];
	}


	// these two can be merged
	for(i=0;i<n;++i){
		t[i] = (fabs(diagD[i]) > TOL) ? 1:0;
	}	

	for(i=0;i<n;++i){
		diagD[i] = (t[i]==1)? diagD[i]:0.f;	
	}

	// p should always be 0

	// update D with diagD
	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			D[i][j] = (i==j)? sqrt(diagD[i]): 0.f;	
		}	
	}

	float **U_t; // transpose of U
	U_t = (float**)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		U_t[i] = (float*)malloc(sizeof(float)*n);
	}

	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			U_t[j][i] = U[i][j];
		}	
	}

	// matrix multiplication
	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			tmp = 0.0;	
			for(k=0;k<n;++k){
				tmp += D[i][k]*U_t[k][j];
			}
			C[i][j] = tmp;
		}	
	}


	// release
	for(i=0;i<n;++i){
		free(x[i]);
		free(u[i]);
		free(D[i]);
		free(U[i]);
		free(U_t[i]);
	}
	free(x);
	free(u);
	free(D);
	free(U);
	free(U_t);
	free(diagD);
	free(t);

}



void pinv_main(float **R, float **R_pinv, unsigned int D)
{
	// input : R
	// output: R_pinv
	// D: row/col

	// use singular value decomposition to find pseudo inverse of a matrix

	unsigned int i;
	unsigned int n = D;
	float **input;
	input = R;

	float **U;
	U = (float**)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		U[i] = (float*)malloc(sizeof(float)*n);
	}

	float **V;
	V = (float **)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		V[i] = (float*)malloc(sizeof(float)*n);
	}

	// S
	float **S;
	S = (float **)malloc(sizeof(float*)*n);
	for(i=0;i<n;++i){
		S[i] = (float*)malloc(sizeof(float)*n);
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
	unsigned int steps,i,j,k,ind;
	double converge;
	double alpha,beta,gamma,zeta;
	double t,c,s;
	float tmp;

	// copy input to U
	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			U[i][j] = input[i][j];
		}
	}

	// identity matrix V
	//eye(n);
	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			V[i][j] = (i==j)? 1.f: 0.f;
		}
	}

	for(i=0;i<n;++i){
		for(j=0;j<n;++j){
			S[i][j] = 0.f;
		}
	}



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
				X[i][j] = 1e-7;	
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

void Forward(float **B, float **A, float *prior, float **alpha, unsigned int N, unsigned int T, float *likelihood)
{
	// B 	: NxT
	// A	: NxN
	// prior: Nx1
	// alpha: NxT

	unsigned int i,j,k;

	float *A_al=(float*)malloc(sizeof(float)*N); 

	float **A_t = (float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		A_t[i] = (float*)malloc(sizeof(float)*N);
	}

	double tmp;
	double alpha_sum;
	double loglikelihood=0.0;

	// ---------------------
	//	alpha = zeros(N, T);
	// ---------------------
	for(i=0;i<N;++i){
		for(j=0;j<T;++j){
			alpha[i][j] = 0.f;
		}
	}

	// populate the probability forward along the time windows T
	for(i=0;i<T;++i)
	{
		if(i==0)
		{
			for(j=0;j<N;++j)
			{
				alpha[j][i] = B[j][i] * prior[j];
				// printf("%10e\n",alpha[j*T+i]);
			}

		}
		else
		{
			
			// alpha(:, t) = B(:, t) .* (A' * alpha(:, t - 1));
			transpose(A, A_t, N, N);
			for(j=0;j<N;++j)
			{
				tmp = 0.0;
				for(k=0;k<N;++k)
				{
					tmp += A_t[j][k]*alpha[k][i-1];
				}
				A_al[j] = (float)tmp;
			}

			for(j=0;j<N;++j)
			{
				alpha[j][i] = B[j][i] * A_al[j];
				//if(i==1){
				//	printf("%10e\n",alpha[j*T+i]);			
				//}
			}
		
		}	

		// alpha_sum      = sum(alpha(:, t));
		// alpha(:, t)    = alpha(:, t) ./ alpha_sum; 
		
		alpha_sum = 0.0;
		
		for(j=0;j<N;++j)
		{
			alpha_sum += alpha[j][i];
		}
		
		for(j=0;j<N;++j)
		{
			alpha[j][i] /= (float)alpha_sum;
			//if(i==1 || i==0){
			//	printf("%10e\n",alpha[j*T+i]);			
			//}
		}

		loglikelihood += log(alpha_sum); 

	}



	*likelihood = (float)loglikelihood;

	// release
	free(A_al);
	for(i=0;i<N;++i){
		free(A_t[i]);
	}
	free(A_t);
}

void Backward(float **A, float **B, float **beta, unsigned int N, unsigned int T)
{
	// A   :  NxN
	// B   :  NxT
	// beta:  NxT
	unsigned int i,j,k;

	float *beta_B = (float*)malloc(sizeof(float)*N);

	// initialize to be zeros
	init_2d_f(beta,N,T,0.f);

	// beta(:, T) = ones(N, 1);
	for(i=0;i<N;++i){
		beta[i][T-1] = 1.f;	
	}

	//---------------------
	//for t = (T - 1):-1:1
	//    beta(:, t) = A * (beta(:, t + 1) .* B(:, t + 1));
	//    beta(:, t) = beta(:, t) ./ sum(beta(:, t));
	//end
	//---------------------
	double beta_sum;
	double tmp;
	
	//printf("T = %d\n",T);

	for(i = (T-1); i-->0; )
	{ 
		//printf("i = %d\n", i);
		for(j=0;j<N;++j){
			beta_B[j] = beta[j][i+1]*B[j][i+1];	
		}	
		
		beta_sum = 0.0;
		for(j=0;j<N;++j){
			tmp = 0.0;
			for(k=0;k<N;++k){
				tmp += A[j][k] * beta_B[k];
			}
			beta[j][i] = (float)tmp;
			beta_sum += tmp;
		}

		for(j=0;j<N;++j){
			beta[j][i] /= (float)beta_sum;
		}
	
	}


	// release
	free(beta_B);
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



void EM(float **observations, float *prior, float **A, float **mu, float **B, float ***Sigma, float **alpha, float **beta, unsigned int D, unsigned int N, unsigned int T)
{
	//-------------------------
	// observations: DxT
	// A : NxN
	// B : NxT
	// alpha:  NxT
	// beta :  NxT
	//-------------------------


	unsigned int i,j,k;
	double tmp;

	// NxN
	float **xi_sum	= (float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		xi_sum[i] = (float*)malloc(sizeof(float)*N);
	}
	init_2d_f(xi_sum,N,N,0.f);

	// NxT
	float **gamma= (float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		gamma[i] = (float*)malloc(sizeof(float)*T);
	}
	init_2d_f(gamma,N,T,0.f);


	// NxN
	float **alpha_beta_B = (float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		alpha_beta_B[i] = (float*)malloc(sizeof(float)*N);
	}

	// NxN
	float **xi_sum_tmp = (float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		xi_sum_tmp[i] = (float*)malloc(sizeof(float)*N);
	}


	// Nx1
	float *gamma_norm = (float*)malloc(sizeof(float)*N);
	float *beta_B = (float*)malloc(sizeof(float)*N);


	//------------------------------------//
	//	E-step
	//------------------------------------//
	//---------------------
	//for t = 1:(T - 1)
	//    xi_sum      = xi_sum + normalise(A .* (alpha(:, t) * (beta(:, t + 1) .* B(:, t + 1))'));
	//    gamma(:, t) = normalise(alpha(:, t) .* beta(:, t));
	//end
	//---------------------
	for(i=0;i<T-1;++i)
	{
		for(j=0;j<N;++j)
		{
			beta_B[j] = beta[j][i+1] * B[j][i+1];
		} 

		// alpha: NxT
		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				alpha_beta_B[j][k]= alpha[j][i]*beta_B[k];			
			}
		}
		// A.*alpha_beta_B
		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				xi_sum_tmp[j][k] = A[j][k]*alpha_beta_B[j][k];
			}
		}

		// normalise
		normalise_2d_f(xi_sum_tmp,N,N);

		if(i<=2 && 0){
			printf("\n----------");
			printf("\ni = %d",i);
			printf("\n----------");
			printf("\n");
			//check_2d_f(xi_sum_tmp,N,N);
		}

		// update xi_sum
		for(j=0;j<N;++j){
			for(k=0;k<N;++k){
				xi_sum[j][k] += (float)xi_sum_tmp[j][k];
			}
		}

		// alpha * beta
		for(j=0;j<N;++j){
			gamma_norm[j] = alpha[j][i] * beta[j][i];
		} // Nx1 

		normalise_1d_f(gamma_norm,N);

		// update gamma after normalisation
		for(j=0;j<N;++j){
			gamma[j][i]= gamma_norm[j];
		} 
	}

	// gamma(:, T) = normalise(alpha(:, T) .* beta(:, T));
	for(j=0;j<N;++j){
		gamma_norm[j] = alpha[j][T-1] * beta[j][T-1];
	} 
	normalise_1d_f(gamma_norm,N);
	for(j=0;j<N;++j){
		gamma[j][T-1]= gamma_norm[j];
	} 


	// check_2d_f(gamma,N,T);




	//------------------------------------//
	//	M-step
	//------------------------------------//
	float *exp_prior=(float*)malloc(sizeof(float)*N); // Nx1
	for(i=0;i<N;++i){
		exp_prior[i] = gamma[i][0];
	}
	// NxN
	float **exp_A=(float**)malloc(sizeof(float*)*N);
	for(i=0;i<N;++i){
		exp_A[i] = (float*)malloc(sizeof(float)*N);
	}

	//------------------------------------
	//expected_A     = mk_stochastic(xi_sum);
	//------------------------------------
	mk_stochastic_2d_f(xi_sum,N,N,exp_A);
	//check_2d_f(exp_A,N,N);

	//------------------------------------
	//    expected_mu    = zeros(D, N);
	//	  expected_Sigma = zeros(D, D, N);
	//------------------------------------
	// DxDxN
	float ***exp_sigma;
	exp_sigma = (float***)malloc(sizeof(float**)*D); // row
	for(i=0;i<D;++i){
		exp_sigma[i] = (float**) malloc(sizeof(float*)*D); // col
		for(j=0;j<D;++j){
			exp_sigma[i][j] = (float*) malloc(sizeof(float)*N);
			for(k=0;k<N;++k){
				exp_sigma[i][j][k] = 0.f;	
			}
		}	
	}

	//------------------------------------
	//	gamma_state_sum = sum(gamma, 2);
	//------------------------------------
	float *gamma_state_sum	= (float*)malloc(sizeof(float)*N); // Nx1

	for(i=0;i<N;++i){
		tmp = 0.0;
		for(j=0;j<T;++j)
		{
			tmp += gamma[i][j];
		}
		//Set any zeroes to one before dividing to avoid NaN
		if(tmp == 0.0){
			tmp = 1.0;
		}
		gamma_state_sum[i]=(float)tmp;
	}	
	// puts("gamma_state_sum");
	//check_1d_f(gamma_state_sum,N);


	//---------------------------------------------------------------------------------------
	// matlab:
	// for s = 1:N
	//    gamma_observations = observations .* repmat(gamma(s, :), [D 1]);
	//    expected_mu(:, s)  = sum(gamma_observations, 2) / gamma_state_sum(s); 	%  updated mean
	//    expected_Sigma(:, :, s) = symmetrize(gamma_observations * observations' / gamma_state_sum(s) - ...
	//        expected_mu(:, s) * expected_mu(:, s)');	% updated covariance
	// end
	//---------------------------------------------------------------------------------------

	// gamma:NxT
	// observations:TxD (need to think it as DxT to avoid transposing) 
	// gamma_obs: DxT
	float **gamma_obs		=(float**)malloc(sizeof(float*)*D); // DxT
	float **gamma_obs_mul	=(float**)malloc(sizeof(float*)*D); // DxD
	float **exp_mu_mul		=(float**)malloc(sizeof(float*)*D); // DxD
	float **obs_div_mu		=(float**)malloc(sizeof(float*)*D); // DxD

	for(i=0;i<D;++i)
	{
		gamma_obs[i] 		= (float*)malloc(sizeof(float)*T);
		gamma_obs_mul[i] 	= (float*)malloc(sizeof(float)*D);
		exp_mu_mul[i] 		= (float*)malloc(sizeof(float)*D);
		obs_div_mu[i] 		= (float*)malloc(sizeof(float)*D);
	}

	// DxN
	float **exp_mu=(float**)malloc(sizeof(float*)*D);
	for(i=0;i<D;++i){
		exp_mu[i] = (float*)malloc(sizeof(float)*N);
	}

	// TxD
	float **observations_t=(float**)malloc(sizeof(float*)*T);
	for(i=0;i<T;++i){
		observations_t[i] = (float*)malloc(sizeof(float)*D);
	}
	transpose(observations, observations_t,D,T);

	// DxD
	float **gammaob_ob_t=(float**)malloc(sizeof(float*)*D);
	for(i=0;i<D;++i){
		gammaob_ob_t[i] = (float*)malloc(sizeof(float)*D);
	}



	//unsigned int offset;
	//unsigned int start;
	unsigned int n;

	for(k=0;k<N;++k)
	{

		//repmat gamma
		//offset = k*T;
		//start = i*T;

		for(i=0;i<D;++i){ // num of repeats
			for(j=0;j<T;++j){
				gamma_obs[i][j]=gamma[k][j];	
			}	
		}

		// now gamma_obs is DxT
		// dot multiplication with observations
		for(i=0;i<D;++i){
			for(j=0;j<T;++j){
				gamma_obs[i][j] *= observations[i][j];	
			}
		}
		if(k==0 && 0){check_2d_f(gamma_obs,D,T);}

		for(i=0;i<D;++i){
			tmp=0.0;
			for(j=0;j<T;++j){
				tmp += gamma_obs[i][j];	
			}
			exp_mu[i][k] = (float)tmp/gamma_state_sum[k]; 
		}
		if(k==0 && 0){check_2d_f(exp_mu,D,N);}

		// gamma_obs(DxT)  *  observations_t (TxD)
		for(i=0;i<D;++i){ // row
			for(j=0;j<D;++j){ // col
				tmp = 0.0;	
				for(n=0;n<T;++n){
					tmp += gamma_obs[i][n]*observations_t[n][j];
				}
				gammaob_ob_t[i][j] = (float)tmp;
			}
		}
		if(k==0 && 0){check_2d_f(gammaob_ob_t,D,D);}

		for(i=0;i<D;++i){ // row
			for(j=0;j<D;++j){ // col
				gammaob_ob_t[i][j] /= gamma_state_sum[k];
			}
		}



		for(i=0;i<D;++i){// row
			for(j=0;j<D;++j){ // col
				exp_mu_mul[i][j] = exp_mu[i][k] * exp_mu[j][k];
			}
		}
		if(k==0 && 0){check_2d_f(exp_mu_mul,D,D);}

		// gammaob_bo_t  - exp_mu_mul
		for(i=0;i<D;++i){// row
			for(j=0;j<D;++j){ // col
				gammaob_ob_t[i][j] -= exp_mu_mul[i][j];
			}
		}
		if(k==0 && 0){check_2d_f(gammaob_ob_t,D,D);}

		// symmetrize()	
		symmetrize_f(gammaob_ob_t,D);
		if(k==0 && 0){check_2d_f(gammaob_ob_t,D,D);}

		// save to exp_sigma
		for(i=0;i<D;++i){// row
			for(j=0;j<D;++j){ // col
				exp_sigma[i][j][k] = gammaob_ob_t[i][j];
			}
		}

	}

	// check_3d_f(exp_sigma,D,D,N);

	// % Ninja trick to ensure positive semidefiniteness
	for(k=0;k<N;++k)
	{
		for(i=0;i<D;++i)
		{
			for(j=0;j<D;++j)
			{
				if(i==j)	exp_sigma[i][j][k] += 0.01;
			}
		}
	}


	// update: this is important
	//prior = expected_prior;
	for(i=0;i<N;++i){
		prior[i] = exp_prior[i];
	}
    // A     = expected_A;
	for(i=0;i<N;++i){
		for(j=0;j<N;++j){
			A[i][j] = exp_A[i][j];
		}
	}
    // mu    = expected_mu;
	for(i=0;i<D;++i){
		for(j=0;j<N;++j){
			mu[i][j] = exp_mu[i][j];
		}
	}

    //Sigma = expected_Sigma; 

	for(k=0;k<N;++k){
		for(i=0;i<D;++i){
			for(j=0;j<D;++j){
				Sigma[i][j][k] = exp_sigma[i][j][k];
			}
		}
	}


	// release
	for(i=0;i<N;++i){
		free(xi_sum[i]);
		free(gamma[i]);
		free(alpha_beta_B[i]);
		free(xi_sum_tmp[i]);
		free(exp_A[i]);
	}
	free(xi_sum);
	free(gamma);
	free(alpha_beta_B);
	free(xi_sum_tmp);
	free(gamma_norm);
	free(beta_B);

	free(exp_prior);
	free(exp_A);

	for(i=0;i<D;++i){
		free(exp_mu[i]);
		free(gamma_obs[i]);
		free(gamma_obs_mul[i]);
		free(exp_mu_mul[i]);
		free(obs_div_mu[i]);
		free(gammaob_ob_t[i]);
	}
	free(exp_mu);
	// 3D exp_sigma
	for(i=0;i<D;++i){
		for(j=0;j<D;++j){
			free(exp_sigma[i][j]);
		}
	}
	for(i=0;i<D;++i){
		free(exp_sigma[i]);
	}
	free(exp_sigma);
	free(gamma_state_sum);

	free(gamma_obs);
	free(gamma_obs_mul);
	free(exp_mu_mul);
	free(obs_div_mu);

	for(i=0;i<T;++i){
		free(observations_t[i]);
	}
	free(observations_t);
	free(gammaob_ob_t);

}






