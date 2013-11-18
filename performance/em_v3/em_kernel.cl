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
	
	size_t ly_TILE = ly * TILE;

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
		data = lds[ly_TILE] + 
			lds[ly_TILE + 1] + 
			lds[ly_TILE + 2] + 
			lds[ly_TILE + 3] +
			lds[ly_TILE + 4] + 
			lds[ly_TILE + 5] + 
			lds[ly_TILE + 6] + 
			lds[ly_TILE + 7] +
			lds[ly_TILE + 8] + 
			lds[ly_TILE + 9] + 
			lds[ly_TILE + 10] + 
			lds[ly_TILE + 11] +
			lds[ly_TILE + 12] + 
			lds[ly_TILE + 13] + 
			lds[ly_TILE + 14] + 
			lds[ly_TILE + 15]; 
	
		lds[ly_TILE] = ( (data == 0.f) ? 1.f : data );
		//printf("Z1[%d] = %.4e\n", gy,Z1[gy]);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	//
	#pragma unroll
	for(i=0; i<iters; ++i)
	{
		gid = i*TILE + lx + gidy;        
	 	xisum[gid] /= lds[ly_TILE];
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


// kernel 13: 
__kernel void expect_mu_dev (
	__global float *gamma_obs,
	__constant float *gamma_state_sum_current,
	__global float *expect_mu,
	const int T,
	const int D,
	const int iters,
	const int currentState)
{
	// gamma_obs : D x T
	// expect_mu : N x D
	// sum along columns for each row (D)

	// 16 x 16
	size_t gx = get_global_id(0); // col
	size_t gy = get_global_id(1); // row 

	size_t lx = get_local_id(0); // col
	size_t ly = get_local_id(1); // row 

	size_t lid = ly * TILE + lx;

	size_t gid;
	size_t gid_y = gy * T;

	//size_t ly_TILE = ly* TILE;
	size_t ly_TILE = ly<<4;

	__local float lds[256];

	lds[lid] = 0.f;

	int i;
	#pragma unroll
	for(i=0; i<iters; ++i)
	{
		gid = lx + i * TILE + gid_y;        
		lds[lid] += gamma_obs[gid];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(lx == 0)
	{
		float data;
		// sum up 16 columns
		data = lds[ly_TILE] + 
			lds[ly_TILE + 1] + 
			lds[ly_TILE + 2] + 
			lds[ly_TILE + 3] +
			lds[ly_TILE + 4] + 
			lds[ly_TILE + 5] + 
			lds[ly_TILE + 6] + 
			lds[ly_TILE + 7] +
			lds[ly_TILE + 8] + 
			lds[ly_TILE + 9] + 
			lds[ly_TILE + 10] + 
			lds[ly_TILE + 11] +
			lds[ly_TILE + 12] + 
			lds[ly_TILE + 13] + 
			lds[ly_TILE + 14] + 
			lds[ly_TILE + 15]; 
	
		expect_mu[currentState * D + gy] = data/gamma_state_sum_current[0];
		//printf("[%d] = %.4e\n", gy, expect_mu[currentState * D + gy]);
	}
}


// kernel 14:
__kernel void expect_sigma_dev(
	__global float *gamma_obs,
	__global float *observations_t,	
	__constant float *gamma_state_sum_current,
	__constant float *expect_mu_state,
	__global float *expect_sigma_sym,
	const int D,
	const int T)
{
	// (DxT) (TxD) will produce DxD 
	__local float lds_a[64]; // 8x8
	__local float lds_b[64]; // 8x8

	int gx = get_global_id(0); //col
	int gy = get_global_id(1); //row

	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int bx = get_group_id(0); // T/8
	int by = get_group_id(1);

	int nx = get_num_groups(0);


	int Col =  bx * 8 + lx; // global col index for output
	int Row =  by * 8 + ly; // global row index for output

	float sum = 0.f;        

	int m;

	for ( m = 0; m < nx ; ++m)
	{
		lds_a[ly * 8 + lx] = gamma_obs[Row * T + m * 8 + lx];        
		lds_b[ly * 8 + lx] = observations_t[(m * 8 + ly) * D + Col];        

		barrier(CLK_LOCAL_MEM_FENCE);
		
		// matrix mul on LDS
		// a x b
		int kk;
		#pragma unroll
		for ( kk = 0; kk < 8 ; ++kk) 
		{
			sum += lds_a[ly * 8 + kk] * lds_b[kk * 8 + lx];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// sum is the mm result of gamma_obs * obs_t
	// sum * gamma_state_sum(s) - expect_mu(s) * expect_mu(s)'
	expect_sigma_sym[Row * D + Col] = sum/gamma_state_sum_current[0] - expect_mu_state[Row] * expect_mu_state[Col];

//	printf("[%d,%d] = %.4e\n", Row,Col, expect_sigma_sym[Row * D + Col]);
}


// kernel 15: symmetrize
__kernel void update_expect_sigma(
	__global float *expect_sigma,	
	__global float *expect_sigma_sym,
	const int D,
	const int N)
{

}






