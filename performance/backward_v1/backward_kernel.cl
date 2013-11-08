//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

#define TILE 16

__kernel void mat_vec(
	__global float *beta,  // no need to initialize
	__global float *B, // T x B
	__global float *A,
	const int N,
	const uint startPos,
	const uint startPos_pre)
{
	// beta(:, t+1) is intialized to 1s
	// thus compute A * B(:,t+1) only

	__local float lds[256];

	// hint: fetch B to local        
	size_t gx = get_global_id(0); // col  
	size_t gy = get_global_id(1); // row 

	size_t lx = get_local_id(0); // col  
	size_t ly = get_local_id(1); // row 

	size_t m =  get_num_groups(1); // this equal to the column groups
	size_t currentRow = gy * N;
	size_t offset_x;
	size_t start = ly * TILE;

	int i;
	lds[start + lx] = 0.f;

	#pragma unroll
	for(i = 0 ; i<m ; ++i){
		offset_x = i*TILE;
		lds[start + lx]         += A[currentRow + lx + offset_x] * B[startPos_pre + lx + offset_x];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// sum across rows
	if( gx == 0)
	{
		beta[ startPos + gy] = lds[start] + lds[start + 1] + lds[start + 2] + lds[start + 3] 
							+ lds[start + 4]  + lds[start + 5] + lds[start + 6]  + lds[start + 7] 
							+ lds[start + 8]  + lds[start + 9] + lds[start + 10] + lds[start + 11] 
							+ lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 

		//printf("(%d) %.4e\n",gy,  beta[startPos + gy]);
	}
}

__kernel void scale_beta(
		__global float *beta,
		const uint startPos,
		const uint startPos_pre,
		__local volatile float *lds,
		const int N,
		const int blks)
{
	// 256 +  1 
	size_t lid = get_local_id(0);
	size_t gid;
	float data;

	int i;

	// initialize T-1
	#pragma unroll
	for(i=0; i<blks; ++i)
	{
		gid = i*256 + lid;        
		beta[startPos_pre + gid] = 1.f;
	}

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



// beta_dev: combine A * (beta .* B)
__kernel void beta_dev(
	__global float *beta,  // no need to initialize
	__global float *B, // T x B
	__global float *A,
	const int N,
	const uint startPos,
	const uint startPos_pre)
{
	// beta(:, t+1) is intialized to 1s
	// thus compute A * B(:,t+1) only

	__local float lds[256];

	// hint: fetch B to local        
	size_t gx = get_global_id(0); // col  
	size_t gy = get_global_id(1); // row 

	size_t lx = get_local_id(0); // col  
	size_t ly = get_local_id(1); // row 

	size_t m =  get_num_groups(1); // this equal to the column groups
	size_t currentRow = gy * N;
	size_t offset_x;
	size_t start = ly * TILE;

	int i;
	lds[start + lx] = 0.f;

	#pragma unroll
	for(i = 0 ; i<m ; ++i){
		offset_x = i*TILE;
		lds[start + lx]  += A[currentRow + lx + offset_x] * ( B[startPos_pre + lx + offset_x] * beta[startPos_pre + lx + offset_x]) ;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// sum across rows
	if( gx == 0)
	{
		beta[startPos + gy] = lds[start] + lds[start + 1] + lds[start + 2] + lds[start + 3] 
							+ lds[start + 4]  + lds[start + 5] + lds[start + 6]  + lds[start + 7] 
							+ lds[start + 8]  + lds[start + 9] + lds[start + 10] + lds[start + 11] 
							+ lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 

		//printf("(%d) %.4e\n",gy,  out[gy]);
	}
}



