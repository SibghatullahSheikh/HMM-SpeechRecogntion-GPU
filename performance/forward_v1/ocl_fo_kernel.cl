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

//----------------------------------------
// B should be T x N
// prior is vector
// alpha should be T x N either
// localsize = 128 
// globalsize = N , where N is multpile of 64
//----------------------------------------

// run job 2
__kernel void init_alpha(
		__global float *B, // use no contant memory for general purpose
		__global float *prior,
		__global float *alpha,
		__global float *alphasum,
		__global float *alphasum_tmp,
		__local volatile float *lds,
		const int N,
		__global float *lld)
{
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);

	if(gid < N){
		lds[lid] =  alpha[gid] = B[gid] * prior[gid];
	}else{
		lds[lid] = 0.f;
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
		alphasum_tmp[get_group_id(0)] = lds[0];
		//printf("(%d) %.4e\n",lid,  lds[0]);
	}
}


__kernel void reduction(
		__global float *alphasum_tmp,
		__global float *alphasum,
		__global float *lld,
	//	__local volatile float *lds,
		const int blks)
{

	/*
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);

	if(gid < blks){
		lds[lid] =  alphasum_tmp[gid];
	}else{
		lds[lid] = 0.f;
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
		alphasum[0] = lds[0];
		printf("(%d) %.4e\n",lid,  lds[0]);
		lld[0] += log(lds[0]);
	}

	*/
}




__kernel void mv_2(
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
	lds[ly*TILE + lx] = 0.f;

	#pragma unroll
	for(i = 0 ; i<m ; ++i){
		offset_x = i*TILE;
		lds[start + lx]         += At[currentRow + lx + offset_x] * alpha[startPos + lx + offset_x];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// sum across rows
	if( gx == 0)
	{
		out[gy] = lds[start]                 + lds[start + 1] + lds[start + 2]         + lds[start + 3]  + lds[start + 4]  + lds[start + 5]  + lds[start + 6]  + lds[start + 7] + 
			lds[start + 8]         + lds[start + 9] + lds[start + 10]        + lds[start + 11] + lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
	}


}


// B x at_alpha
// localsize 64, globalsize N
__kernel void alpha_dev(
		__global float *B, // use no contant memory for general purpose
		__global float *at_alpha,
		__global float *alpha,
		__global float *alphasum_tmp,
		__local volatile float *lds,
		const int N,
		__global float *lld,
		const uint startPos)
{

	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);

	size_t blks = get_num_groups(0);

	float data;

	//printf("(%d) , %.4e\n",gid,  lld[0]);

	// lds[65]
	lds[lid] =  data = B[startPos + gid] * at_alpha[gid];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid <  32) {lds[lid] += lds[lid +  32];}
	if(lid <  16) {lds[lid] += lds[lid +  16];}
	if(lid <   8) {lds[lid] += lds[lid +   8];}
	if(lid <   4) {lds[lid] += lds[lid +   4];}
	if(lid <   2) {lds[lid] += lds[lid +   2];}
	if(lid <   1) {lds[lid] += lds[lid +   1];}

	if(lid == 0){
		alphasum_tmp[get_group_id(0)] = lds[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(lid == 0)
	{
		int i;
		float sum = 0.f;
		#pragma unroll
		for(i=0;i<blks;++i){
			sum += alphasum_tmp[i]; 		
		}
		//alphasum[0] = sum;	
		lds[64] = sum;

		if(gid == 0){ 
			lld[0] += log(sum);	
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	//printf("(%d) %f,	%.4e\n",gid, data , lld[0]);

	alpha[startPos + gid] = data / lds[64];

}




/*


__kernel void alpha_dev(
		__global float *At,
		__global float *alpha,
		__global float *B,
		__local  float *lds,
		const    int N, 
		const    int T)
{
	// hint: fetch B to local        
	size_t gx = get_global_id(0); // col  
	size_t gy = get_global_id(1); // row 

	size_t lx = get_local_id(0); // col  
	size_t ly = get_local_id(1); // row 

	size_t m =  N/TILE;


	int t;
	for( t = 1 ;  t < T ; ++t)
	{ // go through each time frame (rows)

		float offset = (t-1)*N;
		float currentRow = offset + N;

		lds[ly*TILE + lx] = 0.0;

		// matrix x vector
		int i;
		for(i = 0 ; i<m ; ++i){
			lds[ly*TILE + lx] += At[gy * N + lx + i*TILE] * alpha[offset + lx + i*TILE];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// sum across rows
		if( gx == 0)
		{
			int start = ly * TILE;
			float tmp;
			tmp = lds[start] + lds[start + 1] + lds[start + 2] + lds[start + 3] + lds[start + 4] + lds[start + 5]  + lds[start + 6]  + lds[start + 7] + lds[start + 8] + lds[start + 9] + lds[start + 10] + lds[start + 11] + lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
			// vector multiply
			alpha[currentRow + gy] = tmp * B[currentRow + gy];
		}
	
		barrier(CLK_GLOBAL_MEM_FENCE);
	
	}

}

*/



/*
// sum alpha along each row 
// localsize 16/16, global 16/N
__kernel void sum_alpha(
		__global float *alpha,
		__global float *alphasum,
		const int N)
{

	size_t gx = get_global_id(0); // col  
	size_t gy = get_global_id(1); // row 

	size_t lx = get_local_id(0); // col  
	size_t ly = get_local_id(1); // row 

	size_t m =  N/TILE;

	__local volatile float lds[256];

	size_t currentRow = gy*N;

	lds[ly*TILE + lx] = 0.0;

	int i;
	for(i = 0 ; i<m ; ++i){
		lds[ly*TILE + lx] += alpha[currentRow + lx + i*TILE];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if( gx == 0)
	{
		int start = ly * TILE;
		alphasum[gy] = lds[start] + lds[start + 1] + lds[start + 2] + lds[start + 3] + lds[start + 4] + lds[start + 5]  + lds[start + 6]  + lds[start + 7] + lds[start + 8] + lds[start + 9] + lds[start + 10] + lds[start + 11] + lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
	}

}

*/


/*
// scale alpha
// T should be at least 2
__kernel void scale_lilkelihood(
		__global float *alpha,		
		__global float *alphasum,		
		__global float *lld,
		const int N,
		const int T
		)
{
	// localsize  64 ,  global size N
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);

	size_t currentRow = 0;
	alpha[gid] /= alphasum[0];

	lld[0] = alphasum[0];

	int i;
	for( t = 1 ;  t < T ; ++t)
	{
		// scale along current row	
		currentRow =  t * N;
		alpha[currentRow + gid] /= alphasum[t];

		if(gid == 0){
			lld[t] = alphasum[t]/alphasum[t-1];
		}
	}

}

*/




/*

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

	//barrier(CLK_GLOBAL_MEM_FENCE);
	//if (gid < nstates) {
	//	alpha_d[stride + tPos]  = alpha_d[stride + tPos] / alphasum_d[tPos];
	//}

}

*/


/*
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
	
	else{
	
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
		alphamid_d [ get_group_id(0) ] = sm[0];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (gid < nstates) {
		alpha_d[stride + tPos]  = alpha_d[stride + tPos] / alphasum_d[tPos];
	}

}
*/




/*
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
*/


//__kernel void test(__global float *alphasum_d)
//{
//	size_t gid = get_global_id(0);
//
//	if(gid < 16){
//		//atomic_add(&testsum_d[15], 1.0f);
//		AtomicAdd(&alphasum_d[0], 1.0f);
//	}
// //	testsum_d[gid] += 1;
//
//}


/*

__kernel void scale_alpha_dev()
{	

	// hint: fetch B to local        
	size_t gx = get_global_id(0); // col  
	size_t gy = get_global_id(1); // row 

	size_t lx = get_local_id(0); // col  
	size_t ly = get_local_id(1); // row 

	size_t m =  N/TILE;

	int i;
	lds[ly*TILE + lx] = 0.0;
	for(i = 0 ; i<m ; ++i){
		lds[ly*TILE + lx]  += A[gy * N + lx + i*TILE] * B[lx + i*TILE];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// sum across rows
	if( gx == 0)
	{
		int start = ly * TILE;
		float tmp;
		tmp = lds[start]                 + lds[start + 1] + lds[start + 2]         + lds[start + 3]  + lds[start + 4]  + lds[start + 5]  + lds[start + 6]  + lds[start + 7] + 
			lds[start + 8]         + lds[start + 9] + lds[start + 10]        + lds[start + 11] + lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 

			tmp * B[gy];
	}



}

*/

