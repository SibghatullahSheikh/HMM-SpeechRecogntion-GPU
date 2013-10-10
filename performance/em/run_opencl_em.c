#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/opencl.h>

#include "run_opencl_em.h"
#include "hmm.h"
#include "../../utils/ocl_utils.h"

void run_opencl_backward(HMM *word)
{

	puts("\n=>GPU");

	int i;
	int N = word->nstates;
	int T = word->len;
	float *B = word->b;
	float *A = word->a;

	// gpu timer
	cl_ulong gstart, gend;
	double gpuTime;

	// cpu timer
	//struct timeval cstart;
	//struct timeval cend;
	double cpuTime;

	float *betaB;
	betaB= (float*)malloc(sizeof(float)*N);
	init_1d_f(betaB,N,0.f);

	float *beta; // NxT
	beta = (float*)malloc(sizeof(float)*N*T);
	init_2d_f(beta,N,T,0.f);
	for(i = 0 ; i < N ; ++i){
		beta[i*T + T-1] = 1.f;
	}



	//------------------------------------------------
	//  OpenCL 
	//------------------------------------------------

	int chunks;
	chunks = (N+63)/64;

	cl_int err;

	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program

	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*3);
	
	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*2);    

	// read kernel file
	char *fileName = "backward_kernel.cl";
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

	kernel[0] = clCreateKernel(program, "genbetaB", &err);
	OCL_CHECK(err);
	kernel[1] = clCreateKernel(program, "beta_dev", &err);
	OCL_CHECK(err);
	kernel[2] = clCreateKernel(program, "scale_beta", &err);
	OCL_CHECK(err);

	// allocate memory on device 
	cl_mem A_d      	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem B_d      	= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL);
	cl_mem beta_d		= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL);
	cl_mem betaB_d		= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N,    NULL, NULL);
	cl_mem betasum_int_d= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*chunks,  NULL, NULL);
	cl_mem betasum_d    = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float),  	  NULL, NULL);

	// warm up() device
	float *dummy = (float*)malloc(sizeof(float));
	cl_mem dummy_d= clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float),  NULL, NULL);
	for(i=0;i<50;++i){
		err = clEnqueueWriteBuffer(queue, dummy_d, 		CL_TRUE, 	0, sizeof(float), 	dummy, 		0, NULL, NULL);
	}


	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 		CL_TRUE, 	0, sizeof(float)*N*N, 	A, 		0, NULL, &event[0]);
	OCL_CHECK(err);
	err = clEnqueueWriteBuffer(queue, B_d, 		CL_TRUE, 	0, sizeof(float)*N*T, 	B, 		0, NULL, NULL); 
	OCL_CHECK(err);
	err = clEnqueueWriteBuffer(queue, beta_d, 	CL_TRUE, 	0, sizeof(float)*N*T, 	beta, 	0, NULL, NULL); 
	OCL_CHECK(err);

	//  1st kernel: beta * B 
	size_t local_1 = 64;
	size_t global_1 = chunks*64;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &beta_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &B_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &betaB_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[0], 3, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[0], 4, sizeof(int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// 2nd kernel: A * betaB
	size_t local_2 = 64;
	size_t global_2 = chunks*64;

	err  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &betaB_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[1], 2, sizeof(cl_mem), &beta_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[1], 3, sizeof(cl_mem), &betasum_int_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[1], 4, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[1], 5, sizeof(int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	//err |= clSetKernelArg(kernel[1], 6, sizeof(int), &frame);
	//if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[1], 7, sizeof(float)*64, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}



	// 3nd kernel: beta/sum 
	size_t local_3 = 64;
	size_t global_3 = chunks*64;

	err  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &beta_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &betasum_int_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[2], 2, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[2], 3, sizeof(int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[2], 5, sizeof(float), &chunks);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// time capsule
	int frame;

	for(frame = (T-2) ; frame >= 0; frame--)
	{
		// 1st kernel : beta * B
		err |= clSetKernelArg(kernel[0], 5, sizeof(int), &frame);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[0], 1, NULL, &global_1, &local_1, 0, NULL, NULL);
		OCL_CHECK(err);

		if(frame ==  (T-2) && 0)
		{
			clFinish(queue);
			clEnqueueReadBuffer(queue, betaB_d, CL_TRUE, 0, sizeof(float)*N, betaB, 0, NULL , NULL);
			check_1d_f(betaB, N);	
			exit(1);	
		}


		// 2nd kernel; betaB * A
		err |= clSetKernelArg(kernel[1], 6, sizeof(int), &frame);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, &global_2, &local_2, 0, NULL, NULL);
		OCL_CHECK(err);

		if(frame ==  (T-2) && 0)
		{
			clFinish(queue);
			clEnqueueReadBuffer(queue, beta_d, CL_TRUE, 0, sizeof(float)*N*T, beta, 0, NULL , NULL);
			check_2d_f(beta, N, T);	
			exit(1);	
		}



		// 3rd kernle; scale beta
		err |= clSetKernelArg(kernel[2], 4, sizeof(int), &frame);
		if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel[2], 1, NULL, &global_3, &local_3, 0, NULL, NULL);
		OCL_CHECK(err);

		if(frame ==  (T-2) && 0)
		{
			clFinish(queue);
			clEnqueueReadBuffer(queue, beta_d, CL_TRUE, 0, sizeof(float)*N*T, beta, 0, NULL , NULL);
			check_2d_f(beta, N, T);	
			exit(1);	
		}



	}

	clFinish(queue);

	clEnqueueReadBuffer(queue, beta_d, CL_TRUE, 0, sizeof(float)*N*T, beta, 0, NULL , &event[1]);

	err = clWaitForEvents(1,&event[1]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);

	gpuTime = (double)(gend -gstart)/1000000000.0;

	cpuTime = 0.0; 

	printf("oclTime = %lf (s)\n", gpuTime + cpuTime);

	// check
	//check_2d_f(beta,N,T);



	clReleaseMemObject(A_d);
	clReleaseMemObject(B_d);
	clReleaseMemObject(beta_d);
	clReleaseMemObject(betaB_d);
	clReleaseMemObject(dummy_d);

	clReleaseMemObject(betasum_d);
	clReleaseMemObject(betasum_int_d);



	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	for(i=0;i<3;++i){
		clReleaseKernel(kernel[i]);
	}
	for(i=0;i<2;++i){
		clReleaseEvent(event[i]);
	}


	free(beta);
	free(betaB);
	free(kernelSource);

	free(dummy);

	return;
}




