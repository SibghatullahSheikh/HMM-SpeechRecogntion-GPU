//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

//#pragma OPENCL EXTENSION cl_amd_printf : enable



#define TILE 16

__kernel void transpose(
		__global float *A,
		__global float *At,
		__local  float *lds )
{
	// square matrix
	size_t N  = get_global_size(0);

	size_t gx = get_group_id(0);
	size_t gy = get_group_id(1);

	size_t blks_x = get_num_groups(0);

	// reshuffle blocks
	size_t giy = gx;
	size_t gix = (gx + gy)%blks_x;

	size_t lix = get_local_id(0);
	size_t liy = get_local_id(1);

	// use reshuffled blks to index the reading data
	size_t ix = gix * TILE + lix;
	size_t iy = giy * TILE + liy;

	size_t index_in = ix + iy * N; // [iy][ix]

	// copy from global memory to LDS
	size_t ind = liy * TILE  + lix; //[liy][lix]

	lds[ind]            =   A[index_in];

	barrier(CLK_LOCAL_MEM_FENCE);

	ix = giy * TILE + lix;
	iy = gix * TILE + liy;

	// transpose the index inside LDS
	ind = lix * TILE  + liy; // [lix][liy]

	int index_out = ix  + iy * N ;

	At[index_out]           = lds[ind];
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
		//double tmp;
		
        float tmp = 0.f;
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
		//double tmp = 0.0;
		float tmp = 0.f;
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


