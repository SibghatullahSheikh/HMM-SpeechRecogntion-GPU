#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void diagcov(__global float *observation_t, __global float *cov_cl, const int D, const int T, volatile __local float *sm) 
{
	// sample input is 10x6 
	// blockdim is (16,16)
	// griddim is (1,1)

	// each y thread is the column
	// each x thread is the row
	
	size_t x 		= get_local_id(0);
	size_t y 		= get_local_id(1);

	size_t gx 		= get_global_id(0);
	size_t gy 		= get_global_id(1);

	float mean;
	float data;

	int2 blksize={get_local_size(0),get_local_size(1)};

	//--------------------
	// step 1: find the mean for each column
	//--------------------


	// fetch data from global mem to shared memory[16][16]
	if(gx<T && gy<D){
		sm[x*16+y] = data = observation_t[gx*D+gy];	
	}else{
		sm[x*16+y] = 0.f;
	}

	barrier(CLK_LOCAL_MEM_FENCE); 

	// reduction on shared memory [16][16]
	if(x < 8 && y < 8){
		if (blksize.x >= 16 && blksize.y >= 16) { sm[x*16+y] += sm[(x*16+y)+16*8]; }
		if (blksize.x >=  8 && blksize.y >=  8) { sm[x*16+y] += sm[(x*16+y)+16*4]; }
		if (blksize.x >=  4 && blksize.y >=  4) { sm[x*16+y] += sm[(x*16+y)+16*2]; }
		if (blksize.x >=  2 && blksize.y >=  2) { sm[x*16+y] += sm[(x*16+y)+16*1]; }
	}

	// each thread update its mean according to the column value of sm[][y]
	if(x < T){
		int i;
		for(i=0;i<D;++i){
			if(y == i){
				mean = sm[y]/(float)T;
				break;
			}
		}
	}

	//--------------------
	// step 2: find the cov for each column
	//--------------------

	if(gx<T && gy<D){
		sm[x*16+y] = (data - mean)*(data - mean);	
	}else{
		sm[x*16+y] = 0.f;
	}
	barrier(CLK_LOCAL_MEM_FENCE); 
	// another reduction 
	if(x < 8 && y < 8){
		if (blksize.x >= 16 && blksize.y >= 16) { sm[x*16+y] += sm[(x*16+y)+16*8]; }
		if (blksize.x >=  8 && blksize.y >=  8) { sm[x*16+y] += sm[(x*16+y)+16*4]; }
		if (blksize.x >=  4 && blksize.y >=  4) { sm[x*16+y] += sm[(x*16+y)+16*2]; }
		if (blksize.x >=  2 && blksize.y >=  2) { sm[x*16+y] += sm[(x*16+y)+16*1]; }
	}

	// output result
	if(x == 0 && y < D){
		// current D row summation	
		cov_cl[y*D+y] = sm[y]/(float)(T-1);
	}

	//if(x == 0 && y < D ){
	//	cov_cl[x*D+y] = sm[y];
	//}

}

/*


	size_t blkid	= get_group_id(0); // frame id
	size_t blksize	= get_local_size(0); // frame id


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

*/







