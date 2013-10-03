#include <stdio.h>
#include <stdlib.h>

#include <CL/opencl.h>

#include "ocl_utils.h"

void oclcheck(int err, const char *file, const int linenum)
{
	// check cl.h version 1.1
	if(err != CL_SUCCESS){
		//printf("Error(%d)! file %s: line %d\n", err, file, linenum);
		printf("OpenCL Version 1.1, error:  %d!  %s: line %d\n", err, file, linenum);
		if(err == -1){
			printf("Hint: CL_DEVICE_NOT_FOUND.\n");
		}else if(err == -2){
			printf("Hint: CL_DEVICE_NOT_AVAILABLE.\n");
		}else if(err == -3){
			printf("Hint: CL_COMPILER_NOT_AVAILABLE.\n");
		}else if(err == -4){
			printf("Hint: CL_MEM_OBJECT_ALLOCATION_FAILURE.\n");
		}else if(err == -5){
			printf("Hint: CL_OUT_OF_RESOURCES.\n");
		}else if(err == -6){
			printf("Hint: CL_OUT_OF_HOST_MEMORY.\n");
		}else if(err == -7){
			printf("Hint: CL_PROFILING_INFO_NOT_AVAILABLE.\n");
		}else if(err == -8){
			printf("Hint: CL_MEM_COPY_OVERLAP.\n");
		}else if(err == -9){
			printf("Hint: CL_IMAGE_FORMAT_MISMATCH.\n");
		}else if(err == -10){
			printf("Hint: CL_IMAGE_FORMAT_NOT_SUPPORTED.\n");
		}else if(err == -11){
			printf("Hint: CL_BUILD_PROGRAM_FAILURE.\n");
		}else if(err == -12){
			printf("Hint: CL_MAP_FAILURE.\n");
		}else if(err == -13){
			printf("Hint: CL_MISALIGNED_SUB_BUFFER_OFFSET.\n");
		}else if(err == -14){
			printf("Hint: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST.\n");
		}else if(err == -30){
			printf("Hint: CL_INVALID_VALUE.\n");
		}else if(err == -31){
			printf("Hint: CL_INVALID_DEVICE_TYPE.\n");
		}else if(err == -32){
			printf("Hint: CL_INVALID_PLATFORM.\n");
		}else if(err == -33){
			printf("Hint: CL_INVALID_DEVICE.\n");
		}else if(err == -34){
			printf("Hint: CL_INVALID_CONTEXT.\n");
		}else if(err == -35){
			printf("Hint: CL_INVALID_QUEUE_PROPERTIES.\n");
		}else if(err == -36){
			printf("Hint: CL_INVALID_COMMAND_QUEUE.\n");
		}else if(err == -37){
			printf("Hint: CL_INVALID_HOST_PTR.\n");
		}else if(err == -38){
			printf("Hint: CL_INVALID_MEM_OBJECT.\n");
		}else if(err == -39){
			printf("Hint: CL_INVALID_IMAGE_FORMAT_DESCRIPTOR.\n");
		}else if(err == -40){
			printf("Hint: CL_INVALID_IMAGE_SIZE.\n");
		}else if(err == -41){
			printf("Hint: CL_INVALID_SAMPLER.\n");
		}else if(err == -42){
			printf("Hint: CL_INVALID_BINARY.\n");
		}else if(err == -43){
			printf("Hint: CL_INVALID_BUILD_OPTIONS.\n");
		}else if(err == -44){
			printf("Hint: CL_INVALID_PROGRAM.\n");
		}else if(err == -45){
			printf("Hint: CL_INVALID_PROGRAM_EXECUTABLE.\n");
		}else if(err == -46){
			printf("Hint: CL_INVALID_KERNEL_NAME.\n");
		}else if(err == -47){
			printf("Hint: CL_INVALID_KERNEL_DEFINITION.\n");
		}else if(err == -48){
			printf("Hint: CL_INVALID_KERNEL.\n");
		}else if(err == -49){
			printf("Hint: CL_INVALID_ARG_INDEX.\n");
		}else if(err == -50){
			printf("Hint: CL_INVALID_ARG_VALUE.\n");
		}else if(err == -51){
			printf("Hint: CL_INVALID_ARG_SIZE.\n");
		}else if(err == -52){
			printf("Hint: CL_INVALID_KERNEL_ARGS.\n");
		}else if(err == -53){
			printf("Hint: CL_INVALID_WORK_DIMENSION.\n");
		}else if(err == -54){
			printf("Hint: CL_INVALID_WORK_GROUP_SIZE.\n");
		}else if(err == -55){
			printf("Hint: CL_INVALID_WORK_ITEM_SIZE.\n");
		}else if(err == -56){
			printf("Hint: CL_INVALID_GLOBAL_OFFSET.\n");
		}else if(err == -57){
			printf("Hint: CL_INVALID_EVENT_WAIT_LIST.\n");
		}else if(err == -58){
			printf("Hint: CL_INVALID_EVENT.\n");
		}else if(err == -59){
			printf("Hint: CL_INVALID_OPERATION.\n");
		}else if(err == -60){
			printf("Hint: CL_INVALID_GL_OBJECT.\n");
		}else if(err == -61){
			printf("Hint: CL_INVALID_BUFFER_SIZE.\n");
		}else if(err == -62){
			printf("Hint: CL_INVALID_MIP_LEVEL.\n");
		}else if(err == -63){
			printf("Hint: CL_INVALID_GLOBAL_WORK_SIZE.\n");
		}else if(err == -64){
			printf("Hint: CL_INVALID_PROPERTY.\n");
		}else{
			// nothing
		}

		exit(1);	
	}
}


void printCompilerOutput(cl_program program, cl_device_id device) 
{

	cl_build_status build_status;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);

	if(build_status == CL_SUCCESS) {
		printf("No compilation errors for this device\n");
		return;
	}

	size_t ret_val_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

	char* build_log = NULL;
	build_log = (char *)malloc(ret_val_size+1);
	if(build_log == NULL){
		perror("malloc");
		exit(1);
	}

	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size+1, build_log, NULL);
	build_log[ret_val_size] = '\0';

	printf("Build log:\n %s...\n", build_log);
}




