#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/opencl.h>

#include "run_opencl_fo.h"
#include "../../utils/ocl_utils.h"

void run_opencl_fo(HMM *word)
{
	puts("\n=>GPU");

	int i;
	int N = word->nstates;
	int T = word->len;
	float *B = word->b;
	float *A = word->a;
	float *prior = word->pri;

	// gpu timer
	cl_ulong gstart, gend;
	double gpuTime;

	// cpu timer
	struct timeval cstart;
	struct timeval cend;
	double cpuTime;

	float *At; // NxN
	At = (float*)malloc(sizeof(float)*N*N);

	// initialize for checking
	float *alpha;
	alpha = (float*)malloc(sizeof(float)*N*T);
	init_2d_f(alpha,N,T,0.f);

	float *alphasum; // T x 1
	alphasum = (float*)malloc(sizeof(float)*T);
	init_1d_f(alpha,T,0.f);


	//-------------------------------------------
	// assume the local size is always 64, to 
	// avoid the AtomicAdd() which stops working
	// after 15 loops for 256x256 case,
	// the intermediate size can be known ahead
	//-------------------------------------------
	int chunks;
	chunks = (N+63)/64;
	float *alphamid;
	alphamid = (float*)malloc(sizeof(float)*chunks);




	//float *testsum;
	//	testsum = (float *)malloc(sizeof(float)*16);
	//	for(i=0;i<16;++i){
	//		testsum[i] = 0.f;
	//	}
	//
	//	printf("host\n");	
	//	for(i=0;i<16;++i){
	//		printf("%f ", testsum[i]);
	//	}


	//------------------------------------------------
	//  OpenCL 
	//------------------------------------------------
	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program

	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*4);
	
	//cl_event *event = (cl_event*)malloc(sizeof(cl_event)*10);    // create events for profiling and concurrent cmdqueue
	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*2);    

/*
	cl_event **eventwait = (cl_event**)malloc(sizeof(cl_event*)*2);
	for(i=0;i<2;++i){
		eventwait[i] = (cl_event*)malloc(sizeof(cl_event)*2);
	}
	*/




	cl_int err;

	// read kernel file
	char *fileName = "ocl_fo_kernel.cl";
	char *kernelSource;
	size_t size;
	FILE *fh = fopen(fileName, "rb");
	if(!fh) {
		printf("Error: Failed to open kernel file!\n");
		exit(1);
	}
	fseek(fh,0,SEEK_END);
	size=ftell(fh);
	fseek(fh,0,SEEK_SET);
	kernelSource = malloc(size+1);
	size_t result;
	result = fread(kernelSource,1,size,fh);
	if(result != size){ fputs("Reading error", stderr);exit(1);}
	kernelSource[size] = '\0';

	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);
	OCL_CHECK(err);

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	OCL_CHECK(err);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	OCL_CHECK(err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	OCL_CHECK(err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	OCL_CHECK(err);

	// turn on optimization for kernel
	char *options="-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS)
		printCompilerOutput(program, device_id);
	OCL_CHECK(err);

	// 
	kernel[0] = clCreateKernel(program, "transpose", &err);
	OCL_CHECK(err);
	kernel[1] = clCreateKernel(program, "init_alpha", &err);
	OCL_CHECK(err);
	kernel[2] = clCreateKernel(program, "alpha_dev", &err);
	OCL_CHECK(err);
	kernel[3] = clCreateKernel(program, "scale_alpha", &err);
	OCL_CHECK(err);
//	kernel[3] = clCreateKernel(program, "test", &err);
//	OCL_CHECK(err);


	// allocate memory on device 
	cl_mem A_d      	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem At_d     	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	//cl_mem prior_d  	= clCreateBuffer(context, CL_MEM_READ_ONLY,   sizeof(float)*N,    NULL, NULL);
	//cl_mem B_d      	= clCreateBuffer(context, CL_MEM_READ_ONLY,   sizeof(float)*N*T,  NULL, NULL);
	cl_mem prior_d  	= clCreateBuffer(context, CL_MEM_READ_WRITE,   sizeof(float)*N,    NULL, NULL);
	cl_mem B_d      	= clCreateBuffer(context, CL_MEM_READ_WRITE,   sizeof(float)*N*T,  NULL, NULL);
	cl_mem alpha_d  	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL);
	cl_mem alphasum_d   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*T,  NULL, NULL);
	cl_mem alphamid_d   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*chunks,  NULL, NULL);

	// warm up() device
	float *dummy = (float*)malloc(sizeof(float));
	cl_mem dummy_d= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float),  NULL, NULL);
	for(i=0;i<50;++i){
		err = clEnqueueWriteBuffer(queue, dummy_d, 		CL_TRUE, 	0, sizeof(float), 	dummy, 		0, NULL, NULL);
	}

	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 		CL_TRUE, 	0, sizeof(float)*N*N, 	A, 		0, NULL, &event[0]);
	OCL_CHECK(err);
	//err = clEnqueueWriteBuffer(queue, prior_d, 	CL_TRUE, 	0, sizeof(float)*N, 	prior, 	0, NULL, &event[1]);      
	err = clEnqueueWriteBuffer(queue, prior_d, 	CL_TRUE, 	0, sizeof(float)*N, 	prior, 	0, NULL, NULL);      
	OCL_CHECK(err);
	//err = clEnqueueWriteBuffer(queue, B_d, 		CL_TRUE, 	0, sizeof(float)*N*T, 	B, 		0, NULL, &event[2]); 
	err = clEnqueueWriteBuffer(queue, B_d, 		CL_TRUE, 	0, sizeof(float)*N*T, 	B, 		0, NULL, NULL); 
	OCL_CHECK(err);
	//err = clEnqueueWriteBuffer(queue, alpha_d, 	CL_TRUE, 	0, sizeof(float)*N*T, 	alpha, 	0, NULL, &event[3]); 
	err = clEnqueueWriteBuffer(queue, alpha_d, 	CL_TRUE, 	0, sizeof(float)*N*T, 	alpha, 	0, NULL, NULL); 
	OCL_CHECK(err);
//	err = clEnqueueWriteBuffer(queue, alphasum_d, CL_TRUE, 	0, sizeof(float)*T, 	alphasum, 0, NULL, &event[4]);  
	err = clEnqueueWriteBuffer(queue, alphasum_d, CL_TRUE, 	0, sizeof(float)*T, 	alphasum, 0, NULL, NULL);  
	OCL_CHECK(err);



	//  1st kernel: transpose
	size_t local_1[2];
	size_t global_1[2];
	local_1[0]= 16;
	local_1[1]= 16;
	//global_1[0] = ((N+15)/16)*16; // N
	//global_1[1] = ((N+15)/16)*16; // N
	global_1[0] =  N;
	global_1[1] =  N;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &At_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel[0], 2, sizeof(float)*256, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// 2nd kernel
	// initial alpha:  B x Prior 
	size_t local_2 = 64;
	size_t global_2 = chunks*64;

	err = clSetKernelArg(kernel[1],     0,  sizeof(cl_mem),     &B_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[1],     1,  sizeof(cl_mem),     &prior_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[1],     2,  sizeof(cl_mem),     &alpha_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[1],     3,  sizeof(int),        &N);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[1],     4,  sizeof(int),        &T);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	//err = clSetKernelArg(kernel[1],     5,  sizeof(int),        &frame);
	//if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[1],     6,  sizeof(cl_mem),      &alphamid_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[1],     7,  sizeof(float)*64,   NULL);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}



	// 3rd kernel
	size_t local_3 = 64;
	size_t global_3 = chunks*64;

	err = clSetKernelArg(kernel[2],     0,  sizeof(cl_mem),     &B_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[2],     1,  sizeof(cl_mem),     &At_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[2],     2,  sizeof(cl_mem),     &alpha_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[2],     3,  sizeof(int),        &N);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[2],     4,  sizeof(int),        &T);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	//err = clSetKernelArg(kernel[2],     5,  sizeof(int),        &frame);
	//if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[2],     6,  sizeof(cl_mem),      &alphamid_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[2],     7,  sizeof(float)*64,   NULL);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}


	// 4th kernel
	size_t local_4 = 256;
	size_t global_4 = ((N+255)/256)*256;

	err = clSetKernelArg(kernel[3],     0,  sizeof(cl_mem),     &alpha_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3],     1,  sizeof(cl_mem),      &alphasum_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3],     2,  sizeof(cl_mem),      &alphamid_d);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3],     3,  sizeof(int),        &N);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3],     4,  sizeof(int),        &T);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}
	err = clSetKernelArg(kernel[3],     6,  sizeof(int),        &chunks);
	if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}



	// transpose kernel
	//err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, global_1, local_1, 0, NULL, &event[5] );
	err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, global_1, local_1, 0, NULL, NULL );
	OCL_CHECK(err);


	// time capsule
	int frame;

	for(frame = 0 ; frame < T; ++frame)
	{

		if(frame == 0)
		{ // initialize
			err = clSetKernelArg(kernel[1],     5,  sizeof(int),        &frame);
			if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}

			err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, &global_2, &local_2, 0, NULL, NULL);
//			err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, &global_2, &local_2, 0, NULL, &event[6]);
			OCL_CHECK(err);


			err = clSetKernelArg(kernel[3],     5,  sizeof(int),        &frame);
			if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}

			err = clEnqueueNDRangeKernel(queue, kernel[3], 1, NULL, &global_4, &local_4, 0, NULL, NULL);
			//err = clEnqueueNDRangeKernel(queue, kernel[3], 1, NULL, &global_4, &local_4, 0, NULL, &event[7]);
			OCL_CHECK(err);


			if( 0 )
			{
				clFinish(queue);
				clEnqueueReadBuffer(queue, alpha_d, CL_TRUE, 0, sizeof(float)*N*T, alpha, 0, NULL, NULL);
				clEnqueueReadBuffer(queue, alphasum_d, CL_TRUE, 0, sizeof(float)*T, alphasum, 0, NULL, NULL);

				puts("alpha");
				for(i=0;i<N;++i){
					printf("%f\n", alpha[i*T + frame]);	
				}

				puts("alphasum");
				printf("%f\n", alphasum[frame]);	

			}

		}
		else
		{
		
		
			//	if (frame == 1 && 1)
			//	{
					err = clSetKernelArg(kernel[2],     5,  sizeof(int),        &frame);
					if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}

					err = clEnqueueNDRangeKernel(queue, kernel[2], 1, NULL, &global_3, &local_3, 0, NULL, NULL);
					//err = clEnqueueNDRangeKernel(queue, kernel[2], 1, NULL, &global_3, &local_3, 0, NULL, &event[8]);
					OCL_CHECK(err);


				

					err = clSetKernelArg(kernel[3],     5,  sizeof(int),        &frame);
					if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}

					err = clEnqueueNDRangeKernel(queue, kernel[3], 1, NULL, &global_4, &local_4, 0, NULL, NULL);
					//err = clEnqueueNDRangeKernel(queue, kernel[3], 1, NULL, &global_4, &local_4, 0, NULL, &event[7]);
					OCL_CHECK(err);

/*


				}

				if( frame == 14)
				{
					err = clSetKernelArg(kernel[2],     5,  sizeof(int),        &frame);
					if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}

					err = clEnqueueNDRangeKernel(queue, kernel[2], 1, NULL, &global_3, &local_3, 0, NULL, NULL);
					OCL_CHECK(err);


					err = clSetKernelArg(kernel[3],     5,  sizeof(int),        &frame);
					if(err != 0) { printf("%d\n",err);  OCL_CHECK(err); exit(1);}

					err = clEnqueueNDRangeKernel(queue, kernel[3], 1, NULL, &global_4, &local_4, 0, NULL, NULL);
					OCL_CHECK(err);


					clFinish(queue);
					clEnqueueReadBuffer(queue, alpha_d, CL_TRUE, 0, sizeof(float)*N*T, alpha, 0, NULL, NULL);
					clEnqueueReadBuffer(queue, alphasum_d, CL_TRUE, 0, sizeof(float)*T, alphasum, 0, NULL, NULL);

					puts("alpha");
					for(i=0;i<N;++i){
						printf("%f\n", alpha[i*T + frame]);	
					}

					puts("alphasum");
					printf("%f\n", alphasum[frame]);	

				}

*/
		
		}

	}



	clFinish(queue);

	//clEnqueueReadBuffer(queue, alphasum_d, CL_TRUE, 0, sizeof(float)*T, alphasum, 0, NULL , &event[9]);
	clEnqueueReadBuffer(queue, alphasum_d, CL_TRUE, 0, sizeof(float)*T, alphasum, 0, NULL , &event[1]);

	//err = clWaitForEvents(1,&event[9]);
	err = clWaitForEvents(1,&event[1]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);




	//puts("\nalphasum");
	//check_1d_f(alphasum, T)

	tic(&cstart);
	double log_likelihood = 0.0;
	for(i=0;i<T;++i){
		log_likelihood += log(alphasum[i]);	
	}
	tic(&cend);
	cl_ulong diff = (cend.tv_usec - cstart.tv_usec) + 1000000 * (cend.tv_sec - cstart.tv_sec);

	gpuTime = (double)(gend -gstart)/1000000000.0;
	cpuTime = (double) diff/ 1000000.0; 

	//printf("gpuTime  = %lf(s)\n", gpuTime);
	//printf("cpuTime  = %lf(s)\n", cpuTime);
	printf("oclTime = %lf (s)\n", gpuTime + cpuTime);

	printf("log_likelihood = %lf\n", log_likelihood);




	clReleaseMemObject(A_d);
	clReleaseMemObject(At_d);
	clReleaseMemObject(prior_d);
	clReleaseMemObject(B_d);
	clReleaseMemObject(alpha_d);
	clReleaseMemObject(alphasum_d);
	clReleaseMemObject(alphamid_d);

	clReleaseMemObject(dummy_d);

	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	for(i=0;i<4;++i){
		clReleaseKernel(kernel[i]);
	}

	free(kernelSource);
	free(At);
	free(alpha);
	free(alphasum);
	free(alphamid);

	free(dummy);

	return;
}




//-----------------------------------------------------------
//
//  Utility Functions
//
//-----------------------------------------------------------

void read_config(HMM* word, char **files,int job, int states, int len)
{
	// variables
	int i;
	int N = states;
	int T = len;

	word->nstates = N;
	word->len = T;


	// allocate parameters
	word->b   = (float*)malloc(sizeof(float)*N*T);
	word->a   = (float*)malloc(sizeof(float)*N*N);
	word->pri = (float*)malloc(sizeof(float)*N);

	for(i = 0; i < 3; ++i){
		//printf("%s\n",files[job*3+i]);    
		// read B
		if(i == 0){
			//printf("read B[%d][%d] \t\t", N, T);
			read_b(files[job*3],word,N,T);
			//printf("done!\n");
		}

		// read A
		if(i == 1){
			//printf("read A[%d][%d] \t\t", N, N);
			read_a(files[job*3+1],word,N,N);
			//printf("done!\n");
		}

		// read prior
		if(i == 2){
			//printf("read prior[%d] \t\t", N);
			read_pri(files[job*3+2],word,N);
			//printf("done!\n");
		}

	}

}


int getLineNum(char *file)
{
	int lineNum=0;
	char buffer[BUFSIZE];
	FILE *fr = fopen(file, "rb");
	if(fr == NULL) {
		printf("Can't open %s!\n", file);
		exit(1);
	}else{
		while(fgets(buffer, sizeof(buffer), fr))
		{
			lineNum++;
		}
		fclose(fr);
	}

	return lineNum;
}


void read_pri(char *file ,HMM* word, int len)
{
	int lineNum = 0;
	char buf[100];

	FILE *fr;
	fr = fopen(file, "r");
	if(fr == NULL) {
		printf("Can't open %s!\n", file);
		exit(1);
	}else{
		while(fgets(buf, sizeof(buf), fr) != NULL)
		{
			sscanf(buf,"%f", &word->pri[lineNum]);
			lineNum++;
		}
		fclose(fr);
	}
}

void read_b(char *file , HMM* word, int row ,int col)
{
	int i;
	int lineNum = 0;
	char buf[BUFSIZE];
	char *pch;

	FILE *fr;
	fr = fopen(file, "r");
	if(fr == NULL) {
		printf("Can't open %s!\n", file);
		exit(1);
	}else{
		while(fgets(buf, BUFSIZE, fr) != NULL)
		{
			i = 0;
			pch =  strtok(buf," ");
			while(pch != NULL)
			{
				sscanf(pch,"%f", &word->b[lineNum*col+i]);
				i++;
				pch = strtok (NULL, " ");
			}

			lineNum++;
		}
		//printf("%d \n",lineNum);
		fclose(fr);
	}

}


void read_a(char *file , HMM* word, int row ,int col)
{
	int i;
	int lineNum = 0;
	char buf[BUFSIZE];
	char *pch;

	FILE *fr;
	fr = fopen(file, "r");
	if(fr == NULL) {
		printf("Can't open %s!\n", file);
		exit(1);
	}else{
		while(fgets(buf, BUFSIZE, fr) != NULL)
		{
			i = 0;
			pch =  strtok(buf," ");
			while(pch != NULL)
			{
				sscanf(pch,"%f", &word->a[lineNum*col+i]);
				i++;
				pch = strtok (NULL, " ");
			}

			lineNum++;
		}
		//printf("%d \n",lineNum);
		fclose(fr);
	}

}


void copy_2d(float **from_array, float **to_array, int row, int col)
{
	int i,j;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			to_array[i][j] = from_array[i][j];
		}
	}

}


void check_pri(HMM *word)
{
	int i;
	int len = word->len;

	for(i=0;i<len;++i)
	{
		printf("%10.4e\n", word->pri[i]);
	}
	printf("\n");

}


void check_a(HMM *word)
{
	int i,j;
	int row = word->nstates;
	int col = word->len;

	for(i=0;i<row;++i)
	{
		printf("row(%d)\n",i);
		for(j=0;j<col;++j)
		{
			printf("%10.4e\n",word->a[i*col+j]);

		}
		printf("\n\n");
	}

}


void check_b(HMM *word)
{
	int i,j;
	int row = word->nstates;
	int col = word->len;

	for(i=0;i<row;++i)
	{
		printf("row(%d)\n",i);
		for(j=0;j<col;++j)
		{
			printf("%10.4e\n",word->b[i*col+j]);

		}
		printf("\n\n");
	}

}



void free_hmm(HMM *word)
{
	//int T,i;
	//T = word->len;
	//for(i = 0; i < T ; ++i) {
	//  free(word->a[i]);
	//  free(word->b[i]);
	//}

	free(word->a);
	free(word->b);
	free(word->pri);

}

void start(struct timeval *timer)
{
	gettimeofday(timer, NULL);
}


void end(struct timeval *timer)
{
	struct timeval tend, tdiff;
	gettimeofday(&tend, NULL);
	timeval_diff(&tdiff, timer, &tend);
	printf("Elapsed Time = %ld.%06ld(s)\n",tdiff.tv_sec, tdiff.tv_usec);
}

void timeval_diff(struct timeval *tdiff, struct timeval *t1, struct timeval *t2)
{
	long int diff = (t2->tv_usec + 1000000*t2->tv_sec) - (t1->tv_usec + 1000000*t1->tv_sec);
	tdiff->tv_sec  = diff/1000000;
	tdiff->tv_usec = diff%1000000;
}


void transpose(float *A, float *A_t, int row, int col)
{
	int i,j;
	for(i=0;i<row;++i){
		for(j=0;j<col;++j){
			A_t[j*row + i] = A[i*col + j];
		}
	}

}

void check_2d_f(float *x, int row, int col)
{
	int i,j;

	for(i=0;i<row;++i)
	{
		printf("row(%d)\n",i);
		for(j=0;j<col;++j)
		{
			printf("%10.4e\n",x[i*col+j]);

		}
		printf("\n\n");
	}
}

void check_1d_f(float *x, int len)
{
	int i;
	for(i=0;i<len;++i)
	{
		printf("%10.4e \n",x[i]);

	}
}

void init_2d_f(float *frames, int row, int col, float val)
{
	int i,j;
	for(i=0;i<row;++i)
	{
		for (j=0;j<col;++j)
		{
			frames[i*col + j] = val;
		}
	}
}

void init_1d_f(float *frames, int len, float val)
{
	int i;
	for(i=0;i<len;++i)
	{
		frames[i] = val;
	}
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;
	return (diff<0);
}


void tic(struct timeval *timer)
{
	gettimeofday(timer, NULL);
}


void toc(struct timeval *timer)
{
	struct timeval tv_end, tv_diff;
	gettimeofday(&tv_end, NULL);
	timeval_subtract(&tv_diff, &tv_end, timer);
	printf("cpuTime = %ld.%06ld (s)\n", tv_diff.tv_sec, tv_diff.tv_usec);
}



