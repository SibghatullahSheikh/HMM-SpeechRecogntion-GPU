#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h> // va

#include "feature_extraction.c"
#include "train.c"
#include "utils.c"

static char *config_file_name="";
static char *config_nfft="";

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
	int argi;

	if (argc == 1)
	{
		puts("Please specify an audio file.\nUsage: \"./speech --file filename\"\n");
		exit(1);
	}

	for (argi = 1; argi < argc; ++argi)
	{
		if (!strcmp(argv[argi], "--file"))	
		{
			need_argument(argc, argv,argi);
			config_file_name = argv[++argi];
			continue;
		}

		if (!strcmp(argv[argi], "--nfft"))	
		{
			need_argument(argc, argv,argi);
			config_nfft = argv[++argi];
			continue;
		}

		if (argv[argi][0] == '-')
		{
			fatal("'%s' is not a valid command-line option.\n",argv[argi]);
		}
	}

	printf("file name = %s\n", config_file_name);
	printf("nfft = %s\n", config_nfft);

	/*
	   extract features
	 */
	unsigned int i,j;

	int ret;

	// read audio
	unsigned int lineNum,N;
	FILE *fr;
	lineNum = getLineNumber(config_file_name);
	printf("audio length = %d\n",lineNum);

	float *sound;
	sound = (float*)malloc(sizeof(float)*lineNum);

	N = 0;
	fr = fopen(config_file_name, "rb");
	if(fr == NULL) {
		printf("Can't open %s!\n", config_file_name);
	}else{
		while(!feof(fr))
		{
			ret = fscanf(fr,"%f", &sound[N]); 
			if(ret == 0){
				printf("fscanf error! %s line:%d\n", __FILE__, __LINE__);
				exit(1);
			}
			//printf("lineNum = %d\n", N);
			//printf("sound = %e\n", sound[N]);
			N++;
			if(N == lineNum){
				break;	
			}
		}
	}
	fclose(fr);

	// Use buffer() function to trunk the audio into overlapping frames
	// frames = buffer(sound, framesize, overlap) 
	unsigned int framesize = 128; // equal to NFFT
	unsigned int overlap = 96;
	unsigned int featureDim = 6;	// number of freqs in each signal frame

	unsigned int frameNum = ceil((float)lineNum/(framesize-overlap));
	printf("\nbuffer()\n");
	printf("frame number = %d\n", frameNum);

	float **verify_frames; // framesize x framenumber 
	verify_frames = (float**) malloc(sizeof(float*)*framesize);
	for(i=0;i<framesize;++i)
	{
		verify_frames[i] = (float*) malloc(sizeof(float)*frameNum);
	}

	read_frames("apple01_frames.txt",verify_frames,frameNum,framesize);

	// buffer input data
	float **frames; // framesize x framenumber 
	frames = (float**) malloc(sizeof(float*)*framesize);
	for(i=0;i<framesize;++i)
	{
		frames[i] = (float*) malloc(sizeof(float)*frameNum);
	}
	
	// memset 2d array
	init_2d_f(frames, framesize, frameNum, 0.f);

	buffer(sound, frames, lineNum, framesize, overlap, frameNum);

	// compare frames and verify_frames
	unsigned int count=0;
	for(i=0;i<frameNum;++i){
		//printf("Frame %d:\n",i+1);
		for(j=0;j<framesize;++j){
			// if(i==0)	printf("%e \n", frames[j][i]);	

			//printf("%5e\n", frames[i*framesize+j]);	
			//if(fabs(verify_frames[i+j*frameNum] - frames[i*framesize+j]) > 0.00000001){
			if(fabs(verify_frames[j][i] - frames[j][i]) > 0.00000001){
				printf("Error at frames[%d,%d]\n",i,j);
				count++;
			}

		}
	}

	if(count==0)
	{
		printf("buffer right\n");
	}else
	{
		printf("buffer wrong\n");
	}

	// hamming window coefficients
	// w = hamming(framesize);
	float *window;
	window = (float*) malloc(sizeof(float)*framesize);
	hamming_f(window, framesize);
/*
	for(i=0; i<framesize; ++i)
	{
		printf("%f\n", window[i]);
	}
*/

	float **framefrequencies; // framesize x framenumber 
	framefrequencies = (float**) malloc(sizeof(float*)*featureDim);
	for(i=0;i<featureDim;++i)
	{
		framefrequencies[i] = (float*) malloc(sizeof(float)*frameNum);
	}
	
	init_2d_f(framefrequencies, featureDim, frameNum, 0.f);
	

	unsigned int NFFT;
	NFFT = (unsigned int)pow(2, ceil(log2(framesize)));
	printf("\nNFFT:%d\n", NFFT);

	// extract features
	// frequency peaks
	FeatureExtraction_freqpeaks(frames, framefrequencies, featureDim, framesize, frameNum, window, NFFT);

	// check
/*
	for (i = 0; i < frameNum; ++i)
	{
		for (j = 0; j < featureDim; ++j)
		{
			printf("%10.5f ",framefrequencies[j][i]);
		}
		printf("\n");
	}
*/



	//-----------------------------------------------
	//	Training 
	//-----------------------------------------------
	unsigned int hiddenstates = 3;

	TRAIN(framefrequencies, featureDim, frameNum, hiddenstates);

	//-----------------------------------------------
	//	Testing 
	//-----------------------------------------------






	// end of program 
	// release before exit
	for(i=0;i<framesize;++i){
		free(verify_frames[i]);
		free(frames[i]);
	}

	for(i=0;i<featureDim;++i){
		free(framefrequencies[i]);
	}

	free(verify_frames);
	free(frames);
	free(sound);
	free(window);
	free(framefrequencies);



	return 0;
}
