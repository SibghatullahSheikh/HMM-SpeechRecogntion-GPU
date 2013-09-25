#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>
//#include<sys/time.h>
#include<stdarg.h>

#include <CL/cl.h>

#include "../../ocl_utils/ocl_utils.c"

/*
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

*/

void diag_cov(float *input, unsigned int D, unsigned int T, float *result)
{
	// input is TxD (10x6)
	unsigned int i,j,m;

	float *means;
	means = (float*) malloc(sizeof(float)*D);


	// sum along freature dimesion (D)
	double sum;
	for(i=0;i<D;++i){
		sum = 0.0;
		for(j=0;j<T;++j){
			sum = sum + input[j*D+i]; 
		}
		means[i] = (float)sum/(float)T;
	}


	for(i=0;i<D;++i){	// target row/column are the diagonal values
		sum = 0.0;
		for(m=0;m<T;++m){
			sum =  sum + (input[m*D+i]-means[i])*(input[m*D+i]-means[i]); 	
		}		
		result[i*D+i] = (float)(sum)/(float)(T-1);	
	}

	// free 
	free(means);
}






int main(int argc, char *argv[])
{
	//--------------------------
	// parse command line option
	//--------------------------
	//int argi;

/*
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
*/

	//--------------------------
	// init input 
	//--------------------------
	int i,j;
	int D = 6;
	int T = 10; // speech frames

	float *observation_t;
	observation_t = (float*)malloc(sizeof(float)*T*D); // DxT

	// DxT
	float input[60]=
	{
	8.1472369e-01,2.7849822e-01,9.5716695e-01,7.9220733e-01,6.7873515e-01,7.0604609e-01,6.9482862e-01, 7.6551679e-01,  7.0936483e-01,  1.1899768e-01,
	9.0579194e-01,5.4688152e-01,4.8537565e-01,9.5949243e-01,7.5774013e-01,3.1832846e-02,3.1709948e-01, 7.9519990e-01,  7.5468668e-01,  4.9836405e-01,
	1.2698682e-01,9.5750684e-01,8.0028047e-01,6.5574070e-01,7.4313247e-01,2.7692298e-01,9.5022205e-01, 1.8687260e-01,  2.7602508e-01,  9.5974396e-01,
	9.1337586e-01,9.6488854e-01,1.4188634e-01,3.5711679e-02,3.9222702e-01,4.6171391e-02,3.4446081e-02, 4.8976440e-01,  6.7970268e-01,  3.4038573e-01,
	6.3235925e-01,1.5761308e-01,4.2176128e-01,8.4912931e-01,6.5547789e-01,9.7131781e-02,4.3874436e-01, 4.4558620e-01,  6.5509800e-01,  5.8526775e-01,
	9.7540405e-02,9.7059278e-01,9.1573553e-01,9.3399325e-01,1.7118669e-01,8.2345783e-01,3.8155846e-01, 6.4631301e-01,  1.6261174e-01,  2.2381194e-01,
	};


	for(i=0;i<D;++i){ // row
		for(j=0;j<T;++j){  // col
			observation_t[j*D+i] = input[i*T+j]; 
		}
	}

	// debug
	puts("observations(transpose)=");
	for(i=0;i<T;++i){ // row
		for(j=0;j<D;++j){  // col
			printf("%.4e ", observation_t[i*D+j]); 
		}
		printf("\n");
	}



	float *cov_c;
	cov_c= (float*)malloc(sizeof(float)*D*D);
	memset(cov_c,0,sizeof(float)*D*D);

	// ------------------
	// c version of cov
	// ------------------

	diag_cov(observation_t,D,T,cov_c);

	//debug
	puts("\nc version = ");
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
	memset(cov_cl,0,sizeof(float)*D*D);

	// opencl
	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_kernel kernel;                 // kernel

	size_t globalSize[2], localSize[2];
	cl_int err;
	localSize[0] = 16; 
	localSize[1] = 16;

	globalSize[0] = ((localSize[0]+15)/16)*16; 
	globalSize[1] = ((localSize[1]+15)/16)*16; 

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
	kernel = clCreateKernel(program, "diagcov", &err);
	OCL_CHECK(err);

	// Create the input and output arrays in device memory for our calculation

	cl_mem d_observation_t; 
	cl_mem d_cov_cl;

	d_observation_t = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*D*T, NULL, NULL);
	d_cov_cl		= clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*D*D,	NULL, NULL); 

	// fill buffer 1.2 , not in 1.1




	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_observation_t, CL_TRUE, 0, sizeof(float)*D*T, observation_t, 0, NULL, NULL);
	OCL_CHECK(err);

	err = clEnqueueWriteBuffer(queue, d_cov_cl, CL_TRUE, 0, sizeof(float)*D*D, cov_cl, 0, NULL, NULL);
	OCL_CHECK(err);

	// Set the arguments to our compute kernel
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_observation_t);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_cov_cl);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &D);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);} 

	err |= clSetKernelArg(kernel, 4, sizeof(float)*256, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);} // shared memory


	// Execute the kernel over the entire range of the data set  
	// err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	//err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &events[4]);
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
	OCL_CHECK(err);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	//clEnqueueReadBuffer(queue, d_buffer, CL_TRUE, 0, sizeof(float)*frameSize*frameNum, buffer, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, d_cov_cl, CL_TRUE, 0, sizeof(float)*D*D, cov_cl, 0, NULL, NULL);

	puts("\nGPU result=");
	for(i=0;i<D;++i){
		for(j=0;j<D;++j){
			printf("%.4e ", cov_cl[i*D+j]);
		}
		printf("\n");
	}


	//--------------------------
	// free space 
	//--------------------------
	clReleaseMemObject(d_observation_t);
	clReleaseMemObject(d_cov_cl);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(observation_t);
	free(cov_c);
	free(cov_cl);
	free(kernelSource);


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
