__kernel void buffer_ocl(__global float *signal, __global float *buffer_sig, const unsigned int sigLen, 
						const int frameSize, const int frameNum, const int offset, const int overlap, 
						__local  float* sm) 
{
	// probably no need to use shared memory

	size_t frameid  = get_group_id(0); // frame id
	size_t lid 		= get_local_id(0);
	size_t gid 		= get_global_id(0);

	float window;

	unsigned int startpos = frameid*offset;	

	if((startpos-overlap)>=0 && (startpos-overlap+lid) < sigLen){
		//buffer_sig[gid] = signal[startpos-overlap+lid];
		sm[lid] = signal[startpos-overlap+lid];
	}else{
		//buffer_sig[gid] = 0.f;	
		sm[lid] = 0.f;	
	}

	barrier(CLK_LOCAL_MEM_FENCE); 

	// generating hamming windows for current buffered frame
	window = 0.54f - 0.46f*cos((2.f*M_PI_F*lid)/(float)(frameSize-1));

	buffer_sig[gid] = sm[lid]*window;


}







