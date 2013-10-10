#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void genbetaB( 
		__global float *beta, // N x T
		__global float *B,  // N x T
		__global float *betaB, //N x 1
		const int nstates,
		const int T,
		const int tPos)
{
	size_t gid = get_global_id(0);

	if (gid < nstates ) 
	{
		betaB[gid] = beta[gid*T + tPos+1] * B[gid*T + tPos+1];
	}
}



__kernel void beta_dev(
	__global float *A_d,
	__global float *betaB_d,
	__global float *beta_d,
	__global float *betasum_int_d,
	const int nstates,
	const int T,
	const int tPos,
	__local float *sm)
{
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);
	size_t stride = gid*T;

	int i;
	double tmp;

	if(gid < nstates )
	{
		// accumulate sum	
		tmp = 0.0;
		for(i=0;i< nstates;++i){
			tmp +=  A_d[gid*nstates + i] * betaB_d[i];	
		}

		sm[lid] = beta_d[stride + tPos] = (float) tmp;
	}else{
		sm[lid] = 0.f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// reduction on shared memory
	if( lid < 32 )
	{
		volatile __local float *sm1 = sm;
		sm1[lid] += sm1[lid + 32];
		sm1[lid] += sm1[lid + 16];
		sm1[lid] += sm1[lid +  8];
		sm1[lid] += sm1[lid +  4];
		sm1[lid] += sm1[lid +  2];
		sm1[lid] += sm1[lid +  1];
	
	}

	if(lid == 0) 
	{
		betasum_int_d [get_group_id(0)] = sm[0];
	}
}




__kernel void scale_beta(
	__global float *beta_d,
	__global float *betasum_int_d,
	const int nstates,
	const int T,
	const int tPos,
	const int chunks)
{
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);
	size_t stride = gid*T;

	int i;
	double tmp;

	__local float sm[1];

	// the first thread sum up	
	if(lid == 0)
	{
		// accumulate sum   
		tmp = 0.0;
		for(i=0;i<chunks;++i){
			tmp += betasum_int_d[i];   
		}
		sm[0] = (float)tmp;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (gid < nstates) 
	{
		beta_d[stride + tPos]  /= sm[0];
	}

}

