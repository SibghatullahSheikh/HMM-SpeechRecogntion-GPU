#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>
//#include<sys/time.h>
#include<stdarg.h>

#include <CL/cl.h>

#include "../../ocl_utils/ocl_utils.c"


void fatal(const char *fmt, ...)
{
	va_list va;
	va_start(va, fmt);
	fprintf(stderr, "fatal: ");
	vfprintf(stderr, fmt, va);
	fprintf(stderr, "\n");
	fflush(NULL);
	exit(1);
}


static void need_argument(int argc, char *argv[], int argi)
{
	if (argi == argc -1)
		fatal("option %s requires one argument.\n", argv[argi]);
}


void diag_cov(float *input, unsigned int D, unsigned int T, float *result)
{
	// input is DxT (6x85)
	// but think input as input'(TxD), to avoid transpose

	unsigned int i,j,m;

	float *means;
	means = (float*) malloc(sizeof(float)*D);


	// sum along freature dimesion (D)
	double sum;
	for(i=0;i<D;++i){
		sum = 0.0;
		for(j=0;j<T;++j){
			sum = sum + input[i*T+j]; 
		}
		means[i] = (float)(sum/(float)T);
	}


	// work on the lower triangular part of the matrix
	// then update the upper part

	for(i=0;i<D;++i){	// target row/column are the diagonal values
		sum = 0.0;
		for(m=0;m<T;++m){
			sum =  sum + (input[i*T+m]-means[i])*(input[i*T+m]-means[i]); 	
		}		
		result[i*D+i] = (float)(sum/(float)(T-1));	
	}

	// free 
	free(means);
}






int main(int argc, char *argv[])
{
	//--------------------------
	// parse command line option
	//--------------------------
	int argi;
	int D = 6;
	int T = 100; // speech frames

	srand(time(NULL));


	int featureDim_cmd, T_cmd;

	if (argc == 1)
	{
		puts("Please specify an option.\nUsage: \"./speech --frame N(integer) --feature D(interger)\"\n");
		exit(1);
	}

	for (argi = 1; argi < argc; ++argi)
	{
		if (!strcmp(argv[argi], "--frame"))
		{
			need_argument(argc, argv,argi);
			T_cmd = atoi(argv[++argi]) ;
			continue;
		}

		if (!strcmp(argv[argi], "--feature"))
		{
			need_argument(argc, argv,argi);
			featureDim_cmd = atoi(argv[++argi]) ;
			continue;
		}

		if (argv[argi][0] == '-')
		{
			fatal("'%s' is not a valid command-line option.\n",argv[argi]);
		}
	}

	T = T_cmd;
	D = featureDim_cmd;


	//--------------------------
	// init input 
	//--------------------------
	int i,j;

	float *observation;
	observation = (float*)malloc(sizeof(float)*T*D); // DxT
	for(i=0;i<D;++i){ // row
		for(j=0;j<T;++j){
			observation[i*T+j] = (float)rand()/(float)RAND_MAX;
		}
	}

	float *cov_c;
	cov_c= (float*)malloc(sizeof(float)*D*D);
	memset(cov_c,0,sizeof(float)*D*D);

	// ------------------
	// c version of cov
	// ------------------

	diag_cov(observation,D,T,cov_c);

	//debug
	puts("c version = ");
	for(i=0;i<D;++i){ // row
		for(j=0;j<D;++j){
			printf("%.4e ",cov_c[i*D+j]);
		}
		printf("\n");
	}


	// ------------------
	// OpenCL version of diag cov
	// ------------------

	float *cov_cl;
	cov_cl = (float*)malloc(sizeof(float)*D*D);

	// opencl
	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_kernel kernel;                 // kernel

	size_t globalSize, localSize;
	cl_int err;
	localSize = 256; //256 
	globalSize = D*256; // DxT

	// read kernel file
	char *fileName = "par_diagcov_kernel.cl";
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
	//queue = clCreateCommandQueue(context, device_id,CL_QUEUE_PROFILING_ENABLE, &err);
	OCL_CHECK(err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	OCL_CHECK(err);

	// Build the program executable 
	//err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	char *options="-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS){
		printCompilerOutput(program, device_id);
	}
	OCL_CHECK(err);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "buffer_ocl", &err);
	OCL_CHECK(err);

	// Create the input and output arrays in device memory for our calculation
	cl_mem d_sound; // lineNum 
	cl_mem d_buffer; // framesize*frameNum 
	cl_mem d_fft;

	d_sound	 = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*lineNum,  NULL, NULL);
	d_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*frameSize*frameNum,	NULL, NULL); 
	d_fft	 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float2)*frameSize*frameNum,NULL, NULL); 

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_sound, CL_TRUE, 0, sizeof(float)*lineNum, sound, 0, NULL, NULL);
	OCL_CHECK(err);

	// Set the arguments to our compute kernel
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_sound);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_buffer);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &lineNum);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &frameSize);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &frameNum);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &offset);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 6, sizeof(cl_int), &overlap);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 7, sizeof(float)*(frameSize+1), NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);} // shared memory

	err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_fft);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);} 


	// Execute the kernel over the entire range of the data set  
	// err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	//err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &events[4]);
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	OCL_CHECK(err);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	//clEnqueueReadBuffer(queue, d_buffer, CL_TRUE, 0, sizeof(float)*frameSize*frameNum, buffer, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, d_fft, CL_TRUE, 0, sizeof(cl_float2)*frameSize*frameNum, fft, 0, NULL, NULL);


	for(i=0;i<frameNum;++i){
		printf("frame = %d\n", i);
		for(j=0;j<frameSize;++j){
			printf("%.4e\n", buffer[i*frameSize+j]);
		}
	}


	for(i=0;i<frameNum;++i){
		printf("frame = %d\n", i);
		for(j=0;j<frameSize;++j){
			printf("%.4e + i %.4e\n", fft[i*frameSize+j].x, fft[i*frameSize+j].y );
		}
	}


	//--------------------------
	// free space 
	//--------------------------
	clReleaseMemObject(d_buffer);
	clReleaseMemObject(d_sound);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	*/
	free(observation);
	free(cov_c);
	free(cov_cl);
	//free(kernelSource);
	//free(fft);


		return 0;
}

/*


//err = clGetEventProfilingInfo (events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
//OCL_CHECK(err);

//err = clGetEventProfilingInfo (events[5], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
//OCL_CHECK(err);

//gpuTime = (double) (end-start)/1000000000.0;
//printf("gpu runtime(data transfer + kernel) = %lf seconds\n", gpuTime);	

gettimeofday(&cpu_end,NULL);
double t1 = cpu_start.tv_sec + cpu_start.tv_usec/1000000.0;
double t2 = cpu_end.tv_sec   + cpu_end.tv_usec/1000000.0;
//printf("cpu(pthread) runtime = %lf seconds \n", t2-t1);

printf("total = %lf seconds\n", t2-t1);

for(i=0;i<Len;++i){
printf("[%d] %lf\n",i+1, Y[i]);
}



//printf("GPU Result:\n");
//printArray(Y,Len);

//for(i=0;i<6;i++){
//	clReleaseEvent(events[i]);
//}

// release OpenCL resources

//release host memory
free(X);
 */
