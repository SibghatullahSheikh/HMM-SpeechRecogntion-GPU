#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <CL/opencl.h>

#include "run_opencl_fo.h"

void run_opencl_fo(HMM *word)
{
/*
	//------------------------------------------------
	// 	OpenCL 
	//------------------------------------------------
	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*4);

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
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
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
	kernel[1] = clCreateKernel(program, "alpha_dev", &err);
	OCL_CHECK(err);


	// Create the input and output arrays in device memory for our calculation
	cl_mem A_d  = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem At_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, A_d, CL_TRUE, 0, sizeof(float)*N*N, A, 0, NULL, NULL);
	OCL_CHECK(err);

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[0], 1, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[0], 2, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &At_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// create events for profiling and concurrent cmdqueue
	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*4);

	// 2D transpose
	size_t local_1[2];
	size_t global_1[2];
	local_1[0]= 16;
	local_1[1]= 16;
	global_1[0] = ((N+15)/16)*16;
	global_1[1] = ((N+15)/16)*16;

	err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, global_1, local_1, 0, NULL, &event[0]);
	OCL_CHECK(err);


	if(job == 0 && debug){
		clFinish(queue);
		clEnqueueReadBuffer(queue, At_d, CL_TRUE, 0, sizeof(float)*N*N, A_t, 0, NULL, NULL);
		puts("Transpose A");
		check_2d_f(A_t,N,N);
	}

	// time capsule
	int frame;


	//---------------------------------------	
	// initial alpha:  B x Prior 
	//---------------------------------------	
	cl_mem prior_d  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*N, NULL, NULL);
	cl_mem B_d  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*N*T, NULL, NULL);
	cl_mem alpha_d  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*N*T, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, prior_d, CL_TRUE, 0, sizeof(float)*N, prior, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, B_d, CL_TRUE, 0, sizeof(float)*N*T, B, 0, NULL, NULL);

	err  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &B_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &prior_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[1], 2, sizeof(cl_mem), &alpha_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[1], 3, sizeof(int), 	&N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[1], 4, sizeof(int), 	&T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err |= clSetKernelArg(kernel[1], 5, sizeof(int), 	&frame);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	size_t local_2 = 64;
	size_t global_2 = ((N+63)/64)*64;

	for(frame = 0 ; frame < T; ++frame)
	{

		if(frame == 0){ // initialize
			err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, &global_2, &local_2, 0, NULL, &event[1]);
			OCL_CHECK(err);
			if(job == 0 && 0){
				clEnqueueReadBuffer(queue, alpha_d, CL_TRUE, 0, sizeof(float)*N*T, alpha, 0, NULL, NULL);
				puts("Alpha");
				check_2d_f(alpha,N,T);

			}

		}


	}




	//clReleaseMemObject(x2_d);
	//clReleaseMemObject(y_d);
	clReleaseMemObject(A_d);
	clReleaseMemObject(At_d);
	//clReleaseMemObject(prior_d);
	//clReleaseMemObject(B_d);
	//clReleaseMemObject(alpha_d);

	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	for(i=0;i<4;++i){
		clReleaseKernel(kernel[i]);
	}

	free(kernelSource);

*/



	return;
}
