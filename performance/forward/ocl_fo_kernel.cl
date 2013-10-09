#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

//#pragma OPENCL EXTENSION cl_amd_printf : enable

/*
void AtomicAdd(volatile __global float *source, const float operand)
{
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;

	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;

	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
*/

__kernel void transpose( 
		__global float *a,
		const int row,
		const int col,
		__global float *a_t)
{
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);

	if (x < row && y < col ) {
		a_t[y * row + x]= a[x * col + y] ;
	}
}


__kernel void init_alpha(
		__global float *b_d, // use no contant memory for general purpose
		__global float *pi_d,
		__global float *alpha_d,
		const int nstates,
		const int T,
		int tPos,
		__global float *alphamid_d,
		__local float *sm)
{
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);
	size_t stride = gid*T;

	if (gid < nstates) {
		sm[lid] = alpha_d[tPos + stride] = b_d[tPos + stride] * pi_d[gid];
	}else{
		sm[lid] = 0.f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// scaling
	if(lid<32){
		volatile __local float *sm1 = sm;
		sm1[lid] += sm1[lid + 32];
		sm1[lid] += sm1[lid + 16];
		sm1[lid] += sm1[lid +  8];
		sm1[lid] += sm1[lid +  4];
		sm1[lid] += sm1[lid +  2];
		sm1[lid] += sm1[lid +  1];
	}

	if(lid == 0) {
		//AtomicAdd(&alphasum_d[tPos], sm[0]);
		alphamid_d [get_group_id(0) ] = sm[0];
	}
/*
	barrier(CLK_GLOBAL_MEM_FENCE);


	if (gid < nstates) {
		alpha_d[stride + tPos]  = alpha_d[stride + tPos] / alphasum_d[tPos];
	}
*/

}


__kernel void alpha_dev(
		__global float *b_d, // same reason 
		__global float *at_d,
		__global float *alpha_d,
				const int nstates,
				const int T,
				int tPos,
		__global float *alphamid_d,
		__local  float *sm)
{
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);
	size_t stride = gid*T;

	if (gid < nstates) 
	{
		int i;
		double tmp;
		tmp = 0.0;
		for(i = 0; i < nstates; ++i){
			tmp += at_d[ gid*nstates + i] * alpha_d[i*T + tPos-1]; // opt: alpha_d can be in constant ?
		}

		//sm[lid] = alpha_d[stride + tPos] = tmp * b_d[stride + tPos];
		alpha_d[stride + tPos] = tmp * b_d[stride + tPos];

	}
	
/*	
	else{
	
		sm[lid] = 0.f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
*/
/*
	// scaling
	if(lid<32){
		volatile __local float *sm1 = sm;
		sm1[lid] += sm1[lid + 32];
		sm1[lid] += sm1[lid + 16];
		sm1[lid] += sm1[lid +  8];
		sm1[lid] += sm1[lid +  4];
		sm1[lid] += sm1[lid +  2];
		sm1[lid] += sm1[lid +  1];
	}

	if(lid == 0) {
		alphamid_d [ get_group_id(0) ] = sm[0];
	}
*/
/*
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (gid < nstates) {
		alpha_d[stride + tPos]  = alpha_d[stride + tPos] / alphasum_d[tPos];
	}
*/

}



__kernel void scale_alpha(
	__global float *alpha_d,
	__global float *alphasum_d,
	__global float *alphamid_d,
	const int nstates,
	const int T,
	int tPos,
	const int chunks)
{
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);
	size_t stride = gid*T;


	// this is non-efficient, but better than transfering data back to host
	__local float sum[1];

	if(lid == 0 )
	{
		// accumulate sum	
		int i;
		double tmp = 0.0;
		for(i=0;i<chunks;++i){
			tmp += alphamid_d[i];	
		}
		sum[0] = (float)tmp;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(gid == 0)
	{
		alphasum_d[tPos] = sum[0];
	}


	if (gid < nstates) 
	{
		alpha_d[stride + tPos]  = alpha_d[stride + tPos] / sum[0];
	}

}

/*

__kernel void test(__global float *alphasum_d)
{
	size_t gid = get_global_id(0);

	if(gid < 16){
		//atomic_add(&testsum_d[15], 1.0f);
		AtomicAdd(&alphasum_d[0], 1.0f);
	}
 //	testsum_d[gid] += 1;

}

*/


