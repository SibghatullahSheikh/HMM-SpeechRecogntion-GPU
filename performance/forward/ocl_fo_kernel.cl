#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable


inline void AtomicAdd(volatile __global float *source, const float operand)
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


__kernel void init_alpha(
		__global const float *b_d,
		__global const float *pi_d,
		__global float *alpha_d,
		const int nstates,
		const int T,
		const int tPos,
		__global float *alphasum_d,
		volatile __local float *sm)
{
	unsigned int gid = get_global_id(0);
	unsigned int lid = get_local_id(0);
	unsigned int stride = gid*T;

	if (gid < nstates) {
		sm[lid] = alpha_d[tPos + stride] = b_d[tPos + stride] * pi_d[gid];
	}else{
		sm[lid] = 0.f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// scaling
	if(lid<32){
		sm[lid] += sm[lid + 32];
		sm[lid] += sm[lid + 16];
		sm[lid] += sm[lid +  8];
		sm[lid] += sm[lid +  4];
		sm[lid] += sm[lid +  2];
		sm[lid] += sm[lid +  1];
	}

	if(lid == 0) {
		AtomicAdd(&alphasum_d[tPos], sm[0]);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (gid < nstates) {
		alpha_d[stride + tPos]  /= alphasum_d[tPos];
	}

}


__kernel void alpha_dev(
		__global const float *b_d, // read-only
		__global const float *at_d,
		__global float *alpha_d,
		const int nstates,
		const int T,
		const int tPos,
		__global float *alphasum_d,
		volatile __local float *sm)
{
	unsigned int gid = get_global_id(0);
	unsigned int lid = get_local_id(0);
	unsigned int stride = gid*T;
	int i;
	double tmp;

	if ( gid < nstates ) {
		sm[lid] = b_d[stride + tPos];
	}else {
		sm[lid] = 0.f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (gid < nstates) {
		tmp = 0.0;
		for(i = 0; i < nstates; ++i){
			tmp += at_d[gid*nstates+i]*alpha_d[i*T + tPos-1];
		}

		alpha_d[stride + tPos] =  sm[lid] = tmp * sm[lid];
		//data = sm[lid] = tmp * sm[lid];
		//alpha_d[stride + tPos] =  data;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// scaling
	if(lid<32){
		sm[lid] += sm[lid + 32];
		sm[lid] += sm[lid + 16];
		sm[lid] += sm[lid +  8];
		sm[lid] += sm[lid +  4];
		sm[lid] += sm[lid +  2];
		sm[lid] += sm[lid +  1];
	}

	if(lid == 0) {
		AtomicAdd(&alphasum_d[tPos], sm[0]);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (gid < nstates) {
		alpha_d[stride + tPos]  /= alphasum_d[tPos];
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


