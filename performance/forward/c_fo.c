#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h> // va


//static char *config_file_name="";
//static char *config_nfft="";

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



int main(int argc, char*argv[])
{
	// parse command line option
	// -s (number of hidden states)
	// -n (length of observation sequence)

	int argi;

	int nstates = 2;
	int length = 10;

	if (argc == 1)
	{
		puts("Please specify an option.\nUsage: \"./c_fo -s hiddenstates -n length\"\n");
		exit(1);
	}

	for (argi = 1; argi < argc; ++argi)
	{
		if (!strcmp(argv[argi], "-s"))	
		{
			need_argument(argc, argv,argi);
			nstates = atoi(argv[++argi]);
			continue;
		}

		if (!strcmp(argv[argi], "-n"))	
		{
			need_argument(argc, argv,argi);
			length = atoi(argv[++argi]);
			continue;
		}

		if (argv[argi][0] == '-')
		{
			fatal("'%s' is not a valid command-line option.\n",argv[argi]);
		}
	}

	printf("nstates = %d\n", nstates);
	printf("length = %d\n", length);

	// B: observation seqence for hidden states
	// prior: prior probability
	float *B;
	B =(float*) malloc (sizeof(float)*nstates*length);







	free(B);

	return 0;
}
