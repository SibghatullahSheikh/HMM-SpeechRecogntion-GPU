#pragma OPENCL EXTENSIONS cl_khr_fp64 : enable
/*
	alpha(:, t) = B(:, t) .* prior;
*/

__kernel void alpha_dev( 
		__global float *b_d,
		__global float *pi_d,
		__global float *alpha_d,
		const int nstates,
		const int t)
{
	unsigned int gid = get_global_id(0);
	unsigned int offset = t*nstates;

	if (gid < nstates) {
		alpha_d[gid + offset] = pi_d[gid+offset] * b_d[gid];
	}
}


__kernel void transpose( 
		__global float *a,
		const int row,
		const int col,
		__global float *a_t
		)
{
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);

	if (x < row && y < col ) {
		 a_t[y * row + x]= a[x * col + y] ;
	}
}

// single precision general matrix vector multiplitcation
// input: r x c   cx1
// output: r x 1

__kernel void SGEMV(
	__global float *mat,
	const int row,
	const int col,
	__global float *vec,
	__global float *out)
{
	unsigned int gid = get_global_id(0);
	int i;
	double sum = 0.0;
	if(gid < row){
		for(i=0;i<col;++i){
			sum += mat[gid * col + i] * vec[i];	
		}	
		out [gid ] = (float) sum;
	}
}

// 
__kernel void scale_alpha(__global float *alpha, const int nstates, unsigned int length, __global float *log_likelihood)
{
	unsigned int gid = get_global_id(0);
	unsigned int i;

	double sum;

	if(gid < length)
	{
		sum = 0.0;
		for(i = 0; i < nstates; ++i)
		{
			sum += alpha[gid*nstates+i];		
		}	
	
		log_likelihood[gid] = (float) sum;

		for(i = 0; i < nstates; ++i)
		{
			alpha[gid*nstates+i] /= (float)sum;		
		}	
	}
}

// host need to accumulate the results, avoid the atomic operation on device
__kernel void reduction_1d_f (
		__global float *arr, 
		const int len,
		__global float *result,
		__local float *sm)
{
	unsigned int lid =  get_local_id(0);
	unsigned int gid =  get_global_id(0);
	unsigned int blkid = get_group_id(0);

	if(gid < len){
		sm[lid] = arr[gid];	
	}else{
		sm[lid] = 0.f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	
	//
	if(lid<128){
		sm[lid] += sum[lid + 128];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid<64){
		sm[lid] += sum[lid + 64];
	}
	barrier(CLK_LOCAL_MEM_FENCE);


	if(lid<32){
		volatile float *sm1 = sm;
		sm1[lid] += sm1[lid + 32];
		sm1[lid] += sm1[lid + 16];
		sm1[lid] += sm1[lid +  8]; 
		sm1[lid] += sm1[lid +  4]; 
		sm1[lid] += sm1[lid +  2]; 
		sm1[lid] += sm1[lid +  1]; 
	}

	if(lid == 0){
		result[blkid] = sm[0];
	}

}






/*
__global__ void calc_alpha_dev( int nstates,
		float *alpha_t_d,
		float *b_d,
		int obs_t)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < nstates) {
		alpha_t_d[idx] = alpha_t_d[idx] * b_d[(obs_t * nstates) + idx];
	}
}


__global__ void scale_alpha_dev( int nstates, float *alpha_t_d, float scale)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < nstates) {
		alpha_t_d[idx] = alpha_t_d[idx] / scale;
	}
}
*/
