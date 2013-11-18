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
	const int blks,
	const uint tPos_pre)
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


// kernel 8: N
__kernel void gammaT(
	__global float *beta,
	__global float *alpha,
	__global float *gamma,
	const uint tPos)
{
	size_t gid = get_global_id(0);
	size_t id = tPos + gid;

	gamma[id] = alpha[id] * beta[id];	
}


// kernel 9:
__kernel void mk_stochastic(
	__global float *xisum,
	const int iters,
	const int N)
{
	//size_t gx = get_global_id(0);  // col
	size_t gy = get_global_id(1);  // row

	size_t lx = get_local_id(0);  // col
	size_t ly = get_local_id(1);  // row

	size_t lid = ly * TILE + lx;

	size_t gidy = gy * N;
	size_t gid;
	

	__local float lds[256];

	lds[lid] = 0.f;


	int i;
	// fetch data to lds
	#pragma unroll
	for(i=0; i<iters; ++i)
	{
		gid = i*TILE + lx + gidy;        
	 	lds[lid] += xisum[gid];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(lx == 0){
		float data;
		// sum up 16 columns
		data = lds[ly] + lds[ly+1] + lds[ly+2] + lds[ly+3] +
			lds[ly+4] + lds[ly+5] + lds[ly+6] + lds[ly+7] +
			lds[ly+8] + lds[ly+9] + lds[ly+10] + lds[ly+11] +
			lds[ly+12] + lds[ly+13] + lds[ly+14] + lds[ly+15]; 
	
		lds[ly] = ( (data == 0.f) ? 1.f : data );
		//printf("Z1[%d] = %.4e\n", gy,Z1[gy]);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	//
	#pragma unroll
	for(i=0; i<iters; ++i)
	{
		gid = i*TILE + lx + gidy;        
	 	xisum[gid] /= lds[ly];
		//printf("Z1[%d,%d] = %.4e\n", gy,i*TILE + lx,xisum[gid]);
	}

}

// kernel 10: gamma_state_sum, sum along time for each state
__kernel void gamma_state_sum_dev(
	__global float *gamma,
	__global float *gamma_state_sum,
	const int iters,
	const int N,
	const uint stride)
{
	// tpb: 16 x 16 
	// stride =  N * TILE 
	// startPos = gy * stride

	size_t gx = get_global_id(0); // col
	size_t gy = get_global_id(1); // row 

	size_t lx = get_local_id(0); // col
	size_t ly = get_local_id(1); // row 

	size_t startPos = gy * N; // first tb position 

	size_t lid = ly * TILE + lx;
	size_t gid;

	__local float lds[256];

	lds[lid] = 0.f;

	int i;
	#pragma unroll
	for(i=0; i<iters; ++i)
	{
		gid = i * stride + startPos + gx;        
		lds[lid] += gamma[gid];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// reduce 16 x 16 to 16 (column) values
	if(gy == 0)
	{
		//float data;	
		gamma_state_sum[gx]= lds[lx] + 
						lds[lx + TILE] + 
						lds[lx + TILE*2] + 
						lds[lx + TILE*3] + 
						lds[lx + TILE*4] + 
						lds[lx + TILE*5] + 
						lds[lx + TILE*6] + 
						lds[lx + TILE*7] + 
						lds[lx + TILE*8] + 
						lds[lx + TILE*9] + 
						lds[lx + TILE*10] + 
						lds[lx + TILE*11] + 
						lds[lx + TILE*12] + 
						lds[lx + TILE*13] + 
						lds[lx + TILE*14] + 
						lds[lx + TILE*15];

		//gamma_state_sum[gx] = data;
		//printf("[%d] = %.4e\n", gx, data);
	}

}


// kernel 11; copy gamma_tmp
__kernel void copy_currentgamma(
	__global float *gamma,
	__global float *gamma_tmp,
	const int N,
	const int currentCol)
{
	// TxN
	size_t gid = get_global_id(0); // row number 
	size_t id = currentCol + gid * N; // copy along column
	gamma_tmp[gid] = gamma[id];
}


// kernel 12: compute expect_mu
__kernel void gamma_obs_dev(
	__global float *observations,
	__constant float *currentgamma,
	__global float *gamma_obs,
	const int T)
{
	// D x T
	size_t gx = get_global_id(0); // T 
	size_t gy = get_global_id(1); // D 

	size_t lx = get_local_id(0); // T 
	size_t ly = get_local_id(1); // D 

	size_t gid = gy * T + gx;	

	gamma_obs[gid] = observations[gid] * currentgamma[gx];
	//printf("[%d,%d] = %.4e\n", gy,gx, gamma_obs[gid]);
}




/*

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


