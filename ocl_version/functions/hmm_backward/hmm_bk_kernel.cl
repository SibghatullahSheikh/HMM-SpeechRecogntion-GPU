

// beta: nstates x T
__kernel void beta_dev(
		__global float* beta,
		__global float* B,
		const int nstates,
		const int t)
{
	size_t gid = get_global_id(0);
	size_t offset = t*nstates;

	if(gid < states)
	{
		beta[gid + offset] *= B[gid + offset];	
	
	}
}
