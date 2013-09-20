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
            $(error CUDA / AMDAPP toolkit are not installed on Linux!) 
        endif

    endif

else
# windows, osx are not supported
    $(error Windows and OSX are currently not supported, expect Linux!) 
endif

SRCS =  $(wildcard *.c)
PROG =  $(SRCS:.c=)

all: 
	$(CC) $(CCFLAGS) $(CL_INCLUDE) $(CL_LIBPATH) par_buf.c -o par_buf $(CL_FLAGS) $(OPT)

.PHONY: clean
clean:
	rm -rf $(PROG) 
