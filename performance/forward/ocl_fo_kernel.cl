#pragma OPENCL EXTENSIONS cl_khr_fp64 : enable

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

