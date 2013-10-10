CC = gcc
CCFLAGS = -g  -Wall -O3  
CL_FLAGS = -lOpenCL 
OPT = -lm -lrt

UNAME = $(shell uname -s)
#$(info $(UNAME))
OSARCH = $(shell uname -m | grep "_64")

ifeq ($(UNAME),Linux)
    CUDADIR = $(shell ls /usr/local/ | grep cuda )
    AMDAPPDIR = $(shell ls /opt/ | grep AMDAPP )
    ifneq ($(findstring cuda,$(CUDADIR)),)
        CL_INCLUDE = -I/usr/local/cuda/include
        CL_LIBPATH = # /usr/local/cuda/lib
    else 
        ifneq ($(findstring AMDAPP, $(AMDAPPDIR)),)
            CL_INCLUDE = -I/opt/AMDAPP/include
            ifneq ($(findstring _64, $(OSARCH)),)
                CL_LIBPATH = -L/opt/AMDAPP/lib/x86_64 
            else
                CL_LIBPATH = -L/opt/AMDAPP/lib/x86 
            endif

        else
			#$(error CUDA / AMDAPP toolkit are not installed on Linux!) 
        endif

    endif

else
# windows, osx are not supported
    $(error Windows and OSX are currently not supported, expect Linux!) 
endif

SRCS =  $(wildcard *.c)
PROG =  $(SRCS:.c=)

# make all
all: ocl_utils.o hmm.o run_opencl_em.o ocl_em 


#--------------
# build objects
#--------------
ocl_utils.o: ../../utils/ocl_utils.h ../../utils/ocl_utils.c 
	$(CC) $(CCFLAGS) $(CL_INCLUDE) $(CL_LIBPATH) -c ../../utils/ocl_utils.c -o ../../utils/ocl_utils.o $(CL_FLAGS) $(OPT)

hmm.o: hmm.h hmm.c 
	$(CC) $(CCFLAGS) $(CL_INCLUDE) $(CL_LIBPATH) -c hmm.c -o hmm.o $(CL_FLAGS) $(OPT)

run_opencl_em.o: run_opencl_em.h run_opencl_em.c 
	$(CC) $(CCFLAGS) $(CL_INCLUDE) $(CL_LIBPATH) -c run_opencl_em.c -o run_opencl_em.o $(CL_FLAGS) $(OPT)


#--------------
# build programs 
#--------------
ocl_em:  ocl_em.c
	$(CC) $(CCFLAGS) $(CL_INCLUDE) $(CL_LIBPATH) ocl_em.c -o ocl_em  ../../utils/ocl_utils.o hmm.o run_opencl_em.o $(CL_FLAGS) $(OPT)

.PHONY: clean
clean:
	rm -rf $(PROG) ../../utils/ocl_utils.o ./result tmp.txt
