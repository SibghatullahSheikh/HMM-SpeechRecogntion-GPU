//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#pragma OPENCL EXTENSION cl_amd_printf : enable


// opencl 1.2
//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable



//----------------------------------------------------------------------
// atomic add on global memory for float
//----------------------------------------------------------------------
void atomic_add_global(volatile global float *source, const float operand) {
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
	} while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}




#define TILE 16

//----------------------------------------------------------------------
// 1st kernel
//----------------------------------------------------------------------
__kernel void transpose(
		__global float *A,
		__global float *At,
		__local  volatile float *lds,
		const int N)
{
	// square matrix
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




//----------------------------------------------------------------------
// 2nd kernel
//----------------------------------------------------------------------

// B should be T x N
// prior is vector
// alpha should be T x N either
// localsize = 256 
// globalsize = 256 


// run job 2
__kernel void init_alpha(
		__global float *B, // use no contant memory for general purpose
		__global float *prior,
		__global float *alpha,
		__local volatile float *lds,
		const int blks)
{
	// 256 
	//size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);
	size_t gid;
	float data;

	lds[lid] = 0.f;

	// fetch data to lds
	int i;
	#pragma unroll
	for(i=0; i<blks; ++i)
	{
		gid = i*256 + lid;	
		data = B[gid] * prior[gid];
		alpha[gid] = data;
		lds[lid] += data;
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
		//alphasum[0] = lds[0];
		//printf("(%d) %.4e\n",lid,  lds[0]);
		data = lds[0];
	}

	// normalise
	#pragma unroll
	for(i=0; i<blks; ++i)
	{
		gid = i*256 + lid;	
		alpha[gid] /= data;
	}

}


__kernel void mat_vec(
		__global float *At, 
		__global float *alpha, 
		__global float *out, 
		const int N,
		const uint startPos)
{
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
		lds[start + lx]         += At[currentRow + lx + offset_x] * alpha[startPos + lx + offset_x];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// sum across rows
	if( gx == 0)
	{
		out[gy] = lds[start] + lds[start + 1] + lds[start + 2] + lds[start + 3] 
			    + lds[start + 4]  + lds[start + 5] + lds[start + 6]  + lds[start + 7] 
				+ lds[start + 8]  + lds[start + 9] + lds[start + 10] + lds[start + 11] 
				+ lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 

		//printf("(%d) %.4e\n",gy,  out[gy]);
	}

}


// B x at_alpha
// localsize 64, globalsize N
__kernel void alpha_dev(
		__global float *B, // use no contant memory for general purpose
		__global float *at_alpha,
		__global float *alpha,
		__local volatile float *lds,
		const int N,
		const int blks,
		const uint startPos)
{
	// lds[256]
	size_t lid = get_local_id(0);
	float data;
	lds[lid] = 0.f;

	int i;
	#pragma unroll
	for(i=0; i<blks; ++i)
	{
		gid = i * 256 + lid;
		data = B[gid] * at_alpha[gid];
		alpha[startPos + gid] = data;
		lds[lid] += data;
	}
	// lds[256]
	//printf("(%d) , %.4e\n",gid,  alpha[gid]);
	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid < 128) {lds[lid] += lds[lid + 128];}
	if(lid <  64) {lds[lid] += lds[lid +  64];}
	if(lid <  32) {lds[lid] += lds[lid +  32];}
	if(lid <  16) {lds[lid] += lds[lid +  16];}
	if(lid <   8) {lds[lid] += lds[lid +   8];}
	if(lid <   4) {lds[lid] += lds[lid +   4];}
	if(lid <   2) {lds[lid] += lds[lid +   2];}
	if(lid <   1) {lds[lid] += lds[lid +   1];}

	if(lid == 0){
		//alphasum_tmp[get_group_id(0)] = lds[0];
		data = lds[0];
	}

	// normalise
	#pragma unroll
	for(i=0; i<blks; ++i)
	{
		gid = i*256 + lid;	
		alpha[startPos + gid] /= data;
	}

}





