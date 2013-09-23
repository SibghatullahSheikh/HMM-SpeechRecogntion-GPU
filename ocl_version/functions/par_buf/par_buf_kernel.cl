#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void buffer_ocl(__global float *signal, __global float *buffer_sig, const unsigned int sigLen, 
						const int frameSize, const int frameNum, const int offset, const int overlap, 
						__local  float* sm, __global float2 *fft_out) 
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

	//buffer_sig[gid] = sm[lid]*window;

	sm[lid] = sm[lid]*window;

	barrier(CLK_LOCAL_MEM_FENCE); 

	// future: work on raidx2-fft
	// current: only one thread work DFT, where the frameSize = N = NFFT
	//if(lid ==0){
	//	int k,n;
	//	double xr, xi;
	//	for(k=0;k<frameSize;k++){
	//		xr = 0.0;   
	//		xi = 0.0;   
	//		for(n=0;n<frameSize;n++){
	//			xr = xr + sm[n]*cos(2.0*M_PI_F*k*n/frameSize); 
	//			xi = xi - sm[n]*sin(2.0*M_PI_F*k*n/frameSize); 
	//		}
	//		fft_out[k+frameid*frameSize].x = (float)xr;
	//		fft_out[k+frameid*frameSize].y = (float)xi;
	//	}
	//}

	int n;
	double xr, xi;
	xr = 0.0;   
	xi = 0.0;   
	for(n=0;n<frameSize;n++){
		xr = xr + sm[n]*cos(2.0*M_PI_F*(lid)*n/frameSize); 
		xi = xi - sm[n]*sin(2.0*M_PI_F*(lid)*n/frameSize); 
	}

	fft_out[gid] = (float2){(float)xr, (float)xi};

}







