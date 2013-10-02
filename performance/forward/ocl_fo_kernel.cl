#pragma OPENCL EXTENSION cl_khr_fp64 : enable


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


__kernel void alpha_dev( 
		__global const float *b_d,
		__global const float *pi_d,
		__global float *alpha_d,
		const int nstates,
		const int T,
		const int startPos)
{
	unsigned int gid = get_global_id(0);
	unsigned int stride = gid*T;

	if (gid < nstates) {
		alpha_d[startPos + stride] = pi_d[startPos + stride] * b_d[gid];
	}
}

