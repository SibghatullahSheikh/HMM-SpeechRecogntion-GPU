#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "hmm.h"
#include "../../ocl_utils/ocl_utils.c"


extern "C" {         
#include "hmm_fo.h"
}


enum {
	MAX_THREADS_PER_BLOCK = 256
};


char *read_kernel_file(char *fileName)
{
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

	return kernelSource;	
}


float run_hmm_fo(Hmm *in_hmm, Obs *in_obs)
{
	// Host-side variables
	int size;
	int t;
	float log_lik;

	float *scale;
	float *a = in_hmm->a;
	float *b = in_hmm->b;
	float *pi = in_hmm->pi;
	int nsymbols = in_hmm->nsymbols;
	int nstates = in_hmm->nstates;
	int *obs = in_obs->data;
	int length = in_obs->length;

	int err;

	// Allocate host memory
	scale = (float *) malloc(sizeof(float) * length);
	if (scale == 0) {
		fprintf (stderr, "ERROR: Host memory allocation error (scale)\n");
		exit(1);
	}

	
	// --------------------------- opencl --------------------------------//

	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_kernel kernel[3];                 // kernel

	// read kernel file
	char *kernelSource = read_kernel_file("hmm_fo_kernel.cl");


	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	OCL_CHECK(err);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	OCL_CHECK(err);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	OCL_CHECK(err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);
	OCL_CHECK(err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	OCL_CHECK(err);

	// Build the program executable 
	char *options="-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS){
		printCompilerOutput(program, device_id);
	}
	OCL_CHECK(err);

	// Create the compute kernel in the program we wish to run
	kernel[0] = clCreateKernel(program, "init_alpha_dev", &err);
	OCL_CHECK(err);
	//kernel[1] = clCreateKernel(program, "calc_alpha_dev", &err);
	//OCL_CHECK(err);
	//kernel[2] = clCreateKernel(program, "scale_alpha_dev", &err);
	//OCL_CHECK(err);


	// Allocate device memory
	cl_mem a_d; // nstates x nstates
	a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nstates*nstates, NULL, err);	
	OCL_CHECK(err);

	cl_mem b_d;
	b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nstates*nsymbols, NULL, err);	
	OCL_CHECK(err);

	cl_mem pi_d;
	pi_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nstates, NULL, err);	
	OCL_CHECK(err);

	cl_mem obs_d;
	obs_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*length, NULL, err);	
	OCL_CHECK(err);

	cl_mem alpha_d;
	alpha_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nstates*length, NULL, err);	
	OCL_CHECK(err);

	cl_mem alpha_t_d;// nstate alpha values at time t 
	alpha_t_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nstates, NULL, err);	
	OCL_CHECK(err);

	cl_mem ones_d;
	ones_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nstates, NULL, err);	
	OCL_CHECK(err);


	// copy host to device
	err = clEnqueueWriteBuffer(queue, a_d, CL_TRUE, 0, sizeof(float)*nstates*nstates, a, 0, NULL, NULL);
	OCL_CHECK(err);
	err = clEnqueueWriteBuffer(queue, b_d, CL_TRUE, 0, sizeof(float)*nstates*nsymbols, b, 0, NULL, NULL);
	OCL_CHECK(err);
	err = clEnqueueWriteBuffer(queue, pi_d, CL_TRUE, 0, sizeof(float)*nstates, pi, 0, NULL, NULL);
	OCL_CHECK(err);
	err = clEnqueueWriteBuffer(queue, obs_d, CL_TRUE, 0, sizeof(float)*length, obs, 0, NULL, NULL);
	OCL_CHECK(err);



	// Initialize alpha variables 
	localSize = MAX_THREADS_PER_BLOCK;
	globalSize= ((nstates + localSize - 1)/localSize)*localSize;

	// Set the arguments to compute kernel[0]
	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &b_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &pi_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 2, sizeof(int), &nstates);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &alpha_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 4, sizeof(cl_mem), &ones_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 5, sizeof(cl_mem), &obs_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// kernel 0
	err = clEnqueueNDRangeKernel(queue, kernel[0], 1, NULL, globalSize, localSize, 0, NULL, NULL);
	OCL_CHECK(err);

	clFinish(queue);

	// Read the results from the device
	err = clEnqueueReadBuffer(queue, alpha_d, CL_TRUE, 0, sizeof(float)*nstates, alpha, 0, NULL, NULL);
	OCL_CHECK(err);


	// free host memory
	free()


	return log_lik;
}


