//#pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

#define TILE 16

// kernel 1 : vector mul 
// beta x B, at t+1
__kernel void vector_mul(
	__global float *beta,
	__global float *alpha,
	__global float *B,
	__global float *beta_B,
	__global float *gamma_norm,
	const uint tPos,
	const uint tPos_pre)
{
	size_t gid = get_global_id(0);
	beta_B[gid] = beta[tPos + gid] * B[tPos + gid];
	gamma_norm[gid] = alpha[tPos_pre + gid] * beta[tPos_pre + gid];	
	//printf("(%d) %.4e\n", gid , gamma_norm[gid]);
}





// kernel 2 : 2d 
__kernel void state_dev(
	__global float *beta_B,
	__global float *alpha,
	__global float *A,
	__global float *xisum_norm,
	const int N,
	const uint tPos_pre)
{
	// launch 16 x 16 
	size_t gx = get_global_id(0);
	size_t gy = get_global_id(1);

	xisum_norm[gy * N + gx] = alpha[tPos_pre + gy] * beta_B[gx] * A[gy * N +  gx];
	//printf("(%d,%d) %.4e\n",gy,gx, xisum_norm[gy*N + gx]);
}


// normalise on gamma_norm
__kernel void normalise_1d_gamma(
	__global float *gamma_norm,
	__global float *gamma,
	__local volatile float *lds,
	const uint tPos_pre,
	const int blks)
{
	// launch 256 tpb only, do reduction and then scaling
	size_t lid = get_local_id(0);
	size_t gid;

	lds[lid] = 0.f;

	// fetch data to lds
	int i;
	#pragma unroll
	for(i=0; i<blks; ++i)
	{
		gid = i*256 + lid;        
	 	lds[lid] += gamma_norm[gid];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// reduction on 256 
	if(lid < 128) {lds[lid] += lds[lid + 128];}
	if(lid <  64) {lds[lid] += lds[lid +  64];}
	if(lid <  32) {lds[lid] += lds[lid +  32];}
	if(lid <  16) {lds[lid] += lds[lid +  16];}
	if(lid <   8) {lds[lid] += lds[lid +   8];}
	if(lid <   4) {lds[lid] += lds[lid +   4];}
	if(lid <   2) {lds[lid] += lds[lid +   2];}
	if(lid <   1) {lds[lid] += lds[lid +   1];}

	// final sum is located in lds[0]
	barrier(CLK_LOCAL_MEM_FENCE);

	#pragma unroll
	for(i=0; i<blks; ++i)
	{
		gid = i*256 + lid;        
		gamma[tPos_pre + gid] = gamma_norm[gid] / lds[0];

		//float data;
		//gamma[tPos_pre + gid] = data = gamma_norm[gid] / lds[0];
		//printf("(%d) %.4e\n",gid, data);
	}

}


// normalise on xisum_norm 
__kernel void normalise_xisum_s1(
	__global float *xisum_norm,
	__global float *xisum_mid,
	__local volatile float *lds,
	const int N)
{
	// 16 x 16
	size_t gx = get_global_id(0); // col
	size_t gy = get_global_id(1); // row

	size_t lx = get_local_id(0); // col
	size_t ly = get_local_id(1); // row

	size_t bx = get_group_id(0);
	size_t by = get_group_id(1);

	size_t lid = ly * TILE + lx;

	size_t m = get_num_groups(0);

	lds[lid] = xisum_norm[gy * N + gx];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid < 128) {lds[lid] += lds[lid + 128];}
	if(lid <  64) {lds[lid] += lds[lid +  64];}
	if(lid <  32) {lds[lid] += lds[lid +  32];}
	if(lid <  16) {lds[lid] += lds[lid +  16];}
	if(lid <   8) {lds[lid] += lds[lid +   8];}
	if(lid <   4) {lds[lid] += lds[lid +   4];}
	if(lid <   2) {lds[lid] += lds[lid +   2];}
	if(lid <   1) {lds[lid] += lds[lid +   1];}

	if(lid == 0) {		
		xisum_mid[by * m + bx] = lds[0];
	}
}



// second stage: reduction on xisum_mid
__kernel void normalise_xisum_s2(
	__global float *xisum_mid,
	__global float *xisum_fin,
	__local volatile float *lds,
	const uint iters)
{
	// reduce to one final sum
	size_t lid = get_local_id(0);
	size_t gid;
	int i;

	lds[lid] = 0.f;

	// fetch data to lds
	#pragma unroll
	for(i=0; i<iters; ++i)
	{
		gid = i*256 + lid;        
	 	lds[lid] += xisum_mid[gid];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// reduction on 256 
	if(lid < 128) {lds[lid] += lds[lid + 128];}
	if(lid <  64) {lds[lid] += lds[lid +  64];}
	if(lid <  32) {lds[lid] += lds[lid +  32];}
	if(lid <  16) {lds[lid] += lds[lid +  16];}
	if(lid <   8) {lds[lid] += lds[lid +   8];}
	if(lid <   4) {lds[lid] += lds[lid +   4];}
	if(lid <   2) {lds[lid] += lds[lid +   2];}
	if(lid <   1) {lds[lid] += lds[lid +   1];}

	// final sum is located in lds[0]
	if(lid == 0)
	{
		xisum_fin[0] = lds[0];
		//printf("sum = %.4e\n", lds[0]);
	}
}



// third stage: simple scaling
__kernel void normalise_xisum_s3(
	__global float *xisum_norm,
	__global float *xisum_fin,
	const int N)
{
	// NxN 
	size_t gx = get_global_id(0);  // col
	size_t gy = get_global_id(1);  // row

	xisum_norm[gy * N + gx] /= xisum_fin[0];
	//printf("(%d,%d) %.4e\n",gy,gx, xisum_norm[gy * N + gx]);
}


// kernel 7:  add xisum_norm  to xisum 
__kernel void xisum_addon(
	__global float *xisum_norm,
	__global float *xisum,
	const int N)
{
	size_t gx = get_global_id(0);  // col
	size_t gy = get_global_id(1);  // row

	size_t gid = gy * N + gx;

	xisum[gid] += xisum_norm[gid];
}




/*

// kernel3 : normalise xi_sum
__kernel void normalise(
	__global float *xi_sum_norm,
	__global float *xi_sum,
	__local volatile float *lds,
	const int N)
{
	// 256 +  1 

	size_t lid = get_local_id(0);
	size_t gid;
	float data;

	int i;

	lds[lid] = 0.f;

	// work on T-2
	// fetch data to lds
#pragma unroll
	for(i=0; i<blks; ++i)
	{
		gid = i*256 + lid;        
		//data = B[gid] * prior[gid];
		//printf("(%d) %.4e\n",gid,  alpha[gid]);
		lds[lid] += beta[startPos + gid];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// reduction on 256 
	if(lid < 128) {lds[lid] += lds[lid + 128];}
	if(lid <  64) {lds[lid] += lds[lid +  64];}
	if(lid <  32) {lds[lid] += lds[lid +  32];}
	if(lid <  16) {lds[lid] += lds[lid +  16];}
	if(lid <   8) {lds[lid] += lds[lid +   8];}
	if(lid <   4) {lds[lid] += lds[lid +   4];}
	if(lid <   2) {lds[lid] += lds[lid +   2];}
	if(lid <   1) {lds[lid] += lds[lid +   1];}

	if(lid == 0){
		lds[256] = lds[0];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// normalise
#pragma unroll
	for(i=0; i<blks; ++i)
	{
		gid = i*256 + lid;        
		beta[startPos + gid] /= lds[256];

		//printf("(%d) %.4e\n",gid,  beta[startPos + gid]);
	}

}

*/


// kernel4 : normalise gamma_sum 







// kernel 1 : multiply beta and B
//			  alpha * beta

/*

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
		__local float *sma,
		__local float *smb,
		const int D,
		const int T,
		const int k)
{
	// gamma_obs		[Row][m*TILE+ty]
	// observations_t 	[m*TILE+tx][Col] 
	int gx = get_global_id(0); 
	int gy = get_global_id(1);

	int bx = get_group_id(0);
	int by = get_group_id(1);

	int tx = get_local_id(0);
	int ty = get_local_id(1);

	int Row =  bx * TILE + tx;
	int Col =  by * TILE + ty;

	float sum = 0.f;        

	int m,kk;

	for ( m = 0; m < T/TILE ; ++m)
	{
		sma[tx * TILE + ty] = gamma_obs[Row * T + m * TILE + ty];        
		smb[tx * TILE + ty] = observations_t[(m * TILE + tx) * T + Col];        

		barrier(CLK_LOCAL_MEM_FENCE);


		for ( kk = 0; kk < TILE ; ++kk) 
		{
			sum += sma[tx * TILE + kk] * smb[kk * TILE + ty];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	gammaob_ob_t[Row * D + Col] = sum/gamma_state_sum[k];

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

*/


