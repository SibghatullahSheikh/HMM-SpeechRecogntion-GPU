#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void diagcov(__global float *observation, __global float *cov_cl, const int D, const int T, __local float *sm) 
{
	size_t blkid	= get_group_id(0); // frame id
	size_t blksize	= get_local_size(0); // frame id
	size_t lid 		= get_local_id(0);
	size_t gid 		= get_global_id(0);


	float mean;
	float data=0.f;
	//int cur = gid - blkid*blksize;
	// current T is only 85
	if( lid < T){ 
		sm[lid] = data = observation[blkid*T + lid];
	}else{
		//buffer_sig[gid] = 0.f;	
		sm[lid] = 0.f;	
	}

	barrier(CLK_LOCAL_MEM_FENCE); 

	// reduction on shared memory [256]
	if (blksize	>=  256){ 
		if(lid<128){
			sm[lid] += sm[lid + 128]; 
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE); 

	if (blksize	>=  128){ 
		if(lid<64){
			sm[lid] += sm[lid + 64]; 
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE); 

	if(lid< 32){
		volatile float *sm1 = sm;
		if (blksize	>=  64) { sm1[lid] += sm1[lid + 32]; }
		if (blksize	>=  32) { sm1[lid] += sm1[lid + 16]; }
		if (blksize	>=  16) { sm1[lid] += sm1[lid +  8]; }
		if (blksize	>=   8) { sm1[lid] += sm1[lid +  4]; }
		if (blksize	>=   4) { sm1[lid] += sm1[lid +  2]; }
		if (blksize	>=   2) { sm1[lid] += sm1[lid +  1]; }
	}

	if(lid ==0){
		// current D row summation	
		mean = sm[0]/T;
	}


	// next stage
	if( lid < T){ 
		sm[lid] = (data-mean)*(data-mean);
	}else{
		sm[lid] = 0.f;	
	}
	barrier(CLK_LOCAL_MEM_FENCE); 
	// another reduction
	if (blksize	>=  256){ 
		if(lid<128){
			sm[lid] += sm[lid + 128]; 
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE); 

	if (blksize	>=  128){ 
		if(lid<64){
			sm[lid] += sm[lid + 64]; 
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE); 

	if(lid< 32){
		volatile float *sm2 = sm;
		if (blksize	>=  64) { sm2[lid] += sm2[lid + 32]; }
		if (blksize	>=  32) { sm2[lid] += sm2[lid + 16]; }
		if (blksize	>=  16) { sm2[lid] += sm2[lid +  8]; }
		if (blksize	>=   8) { sm2[lid] += sm2[lid +  4]; }
		if (blksize	>=   4) { sm2[lid] += sm2[lid +  2]; }
		if (blksize	>=   2) { sm2[lid] += sm2[lid +  1]; }
	}

	if(lid ==0){
		// current D row summation	
		cov_cl[blkid*D+blkid] = sm[0]/(float)(T-1);
	}

}







