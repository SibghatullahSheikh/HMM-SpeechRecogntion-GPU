#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#pragma OPENCL EXTENSION cl_amd_printf : enable


//---------------------------
// Expectation kernels
//--------------------------

//----------
// opt : memory accessing
//----------

// kernel 1 : multiply beta and B
//			  alpha * beta
__kernel void beta_mul_B(
	__global float *beta,
	__global float *B,
	__global float *alpha,
	__global float *beta_B, 	// 1st output
	__global float *alpha_beta, // 2nd output
	const int N,
	const int T,
	const int window)
{
	size_t gid = get_global_id(0);

	if(gid < N){
		beta_B[gid] 	= beta[gid * T + window + 1] * B[gid * T + window + 1];
		alpha_beta[gid] = alpha[gid * T + window] * beta[gid * T + window];
	}
}


// kernel 2 : A * ( alpha * beta_B )
__kernel void A_alpha_beta_B_dev(
		__global float *beta_B,
		__global float *alpha,
		__global float *A,
		__global float *A_alpha_beta_B,
		const int N,
		const int T,
		const int window)
{
	size_t gx = get_global_id(0);
	size_t gy = get_global_id(1);

	if( gx < N && gy < N )
	{
		A_alpha_beta_B[gx * N + gy] = alpha[gx * T + window] * beta_B[gy] * A[gx * N + gy];			
	}
}



// kernel 3 : 
__kernel void gamma_obs_dev (
		__global float *gamma,
		__global float *observations,
		__global float *gamma_obs,
		const int D,
		const int T,
		const int k)
{
	size_t gx = get_global_id(0); // D
	size_t gy = get_global_id(1); // T

	if( gx < D && gy < T )
	{
		gamma_obs[gx * T + gy] = gamma[k * T + gy] * observations[gx * T + gy];	
	}
}



// kernel 4 :
__kernel void exp_mu_dev (
		__global float *gamma_obs,
		__constant float *gamma_state_sum,
		__global float *exp_mu,
		const int D,
		const int T,
		const int N,
		const int k)
{
	size_t gid = get_global_id(0); // D

	if( gid < D )
	{
		int j;
		double tmp = 0.0;
		for( j = 0; j < T ; ++j ) {
			tmp += gamma_obs[ gid * T + j];	
		}

		exp_mu[gid * N + k] = (float)tmp/gamma_state_sum[k];
	}
}

// kernel 5: matrix multiplication
// 				gamma_obs * observations_t = gammaob_ob_t   ( a * b = c )

__kernel void gammaob_ob_t_dev (
	__global float *gamma_obs,
	__global float *observations_t,
	__global float *gammaob_ob_t,
	__constant float *gamma_state_sum,
	const int D,
	const int T,
	const int k)
{
	size_t gx = get_global_id(0); // i
	size_t gy = get_global_id(1); // j
	int n;

	if( gx < D && gy < D )
	{
		double tmp = 0.0;
		for(n = 0; n < T; ++n) {
			tmp += gamma_obs[gx * T + n] * observations_t[ n * D + gy];
		}

		gammaob_ob_t[gx * D + gy] = (float)tmp / gamma_state_sum[k];

	}

}



// kernel 6 : exp_mu * exp_mu'
__kernel void exp_mu_mul_dev (
	__global float *exp_mu,
	//__global float *exp_mu_mul,
	__global float *gammaob_ob_t,
	const int D,
	const int N,
	const int k)
{
	size_t gx = get_global_id(0); // i
	size_t gy = get_global_id(1); // j

	if( gx < D && gy < D )
	{
		//exp_mu_mul[gx * D + gy] = exp_mu[gx * N + k] * exp_mu[gy * N + k];
		gammaob_ob_t[gx * D + gy] -= (exp_mu[gx * N + k] * exp_mu[gy * N + k]);
	}

}




