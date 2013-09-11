#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "feature_extraction.h"

#define PI  M_PI    /* pi to machine precision, defined in math.h */
#define TWOPI   (2.0*PI)



void buffer(float *sound, float **frames, unsigned int N, unsigned int framesize, unsigned int overlap, unsigned int frameNum)
{
	//double *array_buf = (double*) calloc(frameNum*framesize,sizeof(double));
	unsigned int i,j;
	unsigned int start_pos;
	unsigned int offset=framesize-overlap;

	// at the begining of array, #(overlap) of zeros need to be appeneded before array[]
	// so the pudo array[] will have N+overlap elements
	for(i=0;i<frameNum;i++){
		start_pos=i*offset;
		for(j=0;j<framesize;j++){
			if(start_pos-overlap>=0 && (start_pos-overlap+j)<N){
				frames[j][i] = sound[start_pos-overlap+j];
			}
		}
	}

}

void hamming_f(float *window, unsigned int N)
{
	// Equation 
	// w(n) = 0.54 - 0.46cos(2*pi*n/N), where 0<=n<N
	if(N==1){
		window[0] = 0.f;
	}else{
		unsigned int i;
		for(i=0; i<N; ++i){
			window[i] = 0.54 - 0.46*cos((TWOPI*i)/(float)(N-1));
		}
	}

}

void FFT_F(float *x, unsigned int NFFT, COMPLEX_F *X)
{
	// Equation 
	//          N
	// X(k) =  sum x(n)*exp(-j*2*pi*(k-1)*(n-1)/N),  1<=k<=N
	//         n=1

	unsigned int k,n;
	float xr,xi;

	for(k=0; k < NFFT; ++k)
	{
		xr = xi = 0.f;
		for(n = 0; n < NFFT; ++n)
		{
			xr = xr + x[n]*cos(TWOPI*k*n/(float)NFFT);
			xi = xi - x[n]*sin(TWOPI*k*n/(float)NFFT);
		}

		X[k].real = xr;
		X[k].imag = xi;
	}

}

void BubbleSort(float *arr, unsigned int len)
{
	unsigned int i,j,N;
	float tmp;
	N=len;
	for(i=0;i<N-1;i++){
		for(j=i+1;j<N;j++){
			if(arr[i]<arr[j]){// descending order
				tmp=arr[i];
				arr[i]=arr[j]; 
				arr[j]=tmp;	
			}
		}
	}
}





unsigned int *findpeaks(float *x, unsigned int N, unsigned int *peakNumber)
{
	// find the peaks of x[N]
	unsigned int i,j;

	float *peaks = NULL;
	unsigned int *peaklocation = NULL;

	// dx=x(2:end)-x(1:end-1);
	float *dx;
	dx = (float*)malloc(sizeof(float)*(N-1));

	for( i = 0; i <= (N-2); ++i)
	{
		dx[i] = x[i+1] - x[i];		
	}

	// r=find(dx>0); 
	unsigned int count_r = 0;
	for(i = 0; i <= (N-2); ++i)
	{
		if(dx[i]>0)	count_r=count_r+1;
	}

	unsigned int *r;
	r=(unsigned int*)malloc(sizeof(unsigned int)*count_r);
	count_r=0;
	for(i = 0 ; i < (N-1); ++i)
	{
		if(dx[i]>0)
		{
			r[count_r]=i+1;
			count_r=count_r+1;
		}
	}

	// f=find(dx<0);
	unsigned int count_f=0;
	for(i = 0; i <= (N-2); ++i)
	{
		if(dx[i]<0)	count_f=count_f+1;
	}
	unsigned int *f;
	f=(unsigned int*)malloc(sizeof(unsigned int)*count_f);
	count_f=0;
	for(i = 0; i < (N-1); ++i)
	{
		if(dx[i]<0)
		{
			f[count_f]=i+1;
			count_f=count_f+1;
		}
	}

	if( (count_r>0) & (count_f>0)){
		// dr = r;	
		// dr(2:end)=r(2:end)-r(1:end-1);
		// rc =  ones(nx,1);
		// rc(r+1)=1-dr;
		// rc(1)=0;
		// rs=cumsum(rc); % = time since the last rise
		//printf("\n\n");

		unsigned int *dr;
		dr=(unsigned int*)malloc(sizeof(unsigned int)*count_r);
		dr[0] = r[0];
		for(i=1;i<count_r;i++){
			dr[i]=r[i]-r[i-1];	
		}
		unsigned int *rc;	
		rc=(unsigned int*)calloc(N,sizeof(unsigned int)); // calloc because rc(1) = 0
		for(i=1;i<N;i++){
			rc[i] = 1;	
		}
		for(i=0;i<count_r;i++){
			rc[r[i]]= 1-dr[i]; // indexing difference between matlab and c 
		}

		unsigned int *rs;	
		rs=(unsigned int*)calloc(N,sizeof(unsigned int)); // because rc(1)=0, avoid assigning 0
		for(i=1;i<N;i++){
			rs[i] = rs[i-1]+rc[i];	
		}	

		// df=f;
		// df(2:end)=f(2:end)-f(1:end-1);
		// fc=repmat(1,nx,1);
		// fc(f+1)=1-df;
		// fc(1)=0;
		// fs=cumsum(fc); % = time since the last fall
		unsigned int *df;
		df=(unsigned int*)malloc(sizeof(unsigned int)*count_f);
		df[0] = f[0];
		for(i=1;i<count_f;i++){
			df[i]=f[i]-f[i-1];	
		}
		unsigned int *fc;	
		fc=(unsigned int*)calloc(N,sizeof(unsigned int)); // use calloc to avoid assigning fc(1)=0
		for(i=1;i<N;i++){
			fc[i] = 1;	
		}
		for(i=0;i<count_f;i++){
			fc[f[i]]= 1-df[i]; // different indexing style
		}

		unsigned int *fs;	
		fs=(unsigned int*)calloc(N,sizeof(unsigned int)); // avoid assigning 0
		for(i=1;i<N;i++){
			fs[i] = fs[i-1]+fc[i];	
		}	

		// rp=repmat(-1,nx,1);
		// rp([1; (r+1)'])=[(dr-1)'; nx-r(end)-1];
		// rq=cumsum(rp);  % = time to the next rise
		int *rp;
		rp = (int*)malloc(sizeof(int)*N);
		for(i=0;i<N;i++){
			rp[i]=-1;	
		}
		int *rp_cp;
		rp_cp = (int*)malloc(sizeof(int)*(count_r+1));
		for(i=0;i<count_r;i++){
			rp_cp[i]=dr[i]-1;	
		}
		rp_cp[count_r] = N-1-r[count_r-1];

		rp[0] = rp_cp[0];
		for(i=0;i<count_r;i++){
			rp[r[i]] = rp_cp[i+1]; // different indexing style 	
		}

		int *rq;
		rq = (int*)malloc(sizeof(int)*N);
		rq[0] = rp[0];
		for(i=1;i<N;i++){
			rq[i] = rq[i-1] + rp[i];	
		}

		// fp=repmat(-1,nx,1);
		// fp([1; (f+1)'])=[(df-1)'; nx-f(end)-1];
		// fq=cumsum(fp); % = time to the next fall
		int *fp;
		fp = (int*)malloc(sizeof(int)*N);
		for(i=0;i<N;i++){
			fp[i]=-1;	
		}

		int *fp_cp;
		fp_cp = (int*)malloc(sizeof(int)*(count_f+1));
		for(i=0;i<count_f;i++){
			fp_cp[i]=df[i]-1;	
		}
		fp_cp[count_f] = N-1-f[count_f-1];
		
		fp[0] = fp_cp[0];
		for(i=0;i<count_f;i++){
			fp[f[i]] = fp_cp[i+1];// different indexing style 	
		}

		int *fq;
		fq = (int*)malloc(sizeof(int)*N);
		fq[0] = fp[0];
		for(i=1;i<N;i++){
			fq[i] = fq[i-1] + fp[i];	
		}

		// k=find((rs<fs) & (fq<rq) & (floor((fq-rs)/2)==0));   % the final term centres peaks within a plateau
		// v=x(k);
		char *find_k;
		find_k = (char*)calloc(N,sizeof(char));
		for(i=0;i<N;i++){
			if(rs[i]<fs[i]){
				find_k[i] = 1;	
			}	
		}
		for(i=0;i<N;i++){
			if(fq[i]<rq[i]){

			}else{
				find_k[i]=0;
			}	
		}
		for(i=0;i<N;i++){
			if( floor((fq[i]-rs[i])/2.0) == 0){

			}else{
				find_k[i]=0;
			}	
		}

		unsigned int count_k=0;
		for(i=0;i<N;i++){
			if(find_k[i] != 0){
				count_k = count_k+1;
			}	
		}

		if(count_k ==0){
			*peakNumber = 0;
		}else{
			*peakNumber = count_k;

			peaks = (float*)malloc(sizeof(float)*count_k);
			peaklocation = (unsigned int*)malloc(sizeof(unsigned int)*count_k);

			j=0;

			for(i=0;i<N;i++){
				if(find_k[i] != 0){
					peaks[j] = x[i];
					j =  j + 1;
				}	
			}

			// sort in descending order
			BubbleSort(peaks,count_k);

			// here can be optimized: use less comparisions
			for(i=0;i<count_k;i++){
				for(j=0;j<N;j++){
					if(peaks[i]==x[j]){
						peaklocation[i] = j;	
					}
				}
			}

		}



		free(find_k);
		free(fq);
		free(fp_cp);
		free(fp);
		free(rq);
		free(rp_cp);
		free(rp);
		free(fs);
		free(fc);
		free(df);
		free(rs);
		free(rc);
		free(dr);
	}

	free(f);
	free(r);
	free(dx);

	return peaklocation;
}


void FeatureExtraction_freqpeaks(float **frames, float **framefrequencies, unsigned int featureDim, unsigned int framesize, unsigned int frameNum, float *window, unsigned int NFFT)
{
	// -----------------------------------------------------------------------------------------
	//	 	for frame = frames
	//			x = frame .* w;
	//			X = fft(x, NFFT)/framesize;
	//			f = Fs/2*linspace(0, 1, NFFT/2 + 1);
	//
	//			% Finding local maxima in single-sided amplitude spectrum
	//				[~, peaklocations] = findpeaks(abs(X(1:(NFFT/2 + 1))), 'SORTSTR', 'descend');
	//
	//			if isempty(peaklocations)
	//				peaklocations = ones(D, 1);
	//			elseif length(peaklocations) < D
	//				peaklocations = padarray(peaklocations, D - length(peaklocations), 1, 'post');
	//			end
	//
	//				framefrequencies(:, i) = f(peaklocations(1:D))';
	//			i = i + 1;
	//		end
	// -----------------------------------------------------------------------------------------
	unsigned int i,j;
	unsigned int Fs = 8000;
	double li;

	float *x;
	x =(float*)malloc(sizeof(float)*framesize);

	// f
	unsigned int linnum = NFFT/2+1;
	float *f; // half fft
	f= (float*)malloc(sizeof(float)*linnum);

	float half_fs = Fs*0.5;
	float linstep = 1.0/(float)(linnum-1);
	j=0;
	for ( li = 0; li <= 1; li = li+linstep, j = j+1)
	{
		f[j] = li * half_fs;	
		//printf("f=%e\n",f[j]);
	}

	COMPLEX_F *X;
	X = (COMPLEX_F*) malloc(sizeof(COMPLEX_F)*framesize); 

	float *absX;
	absX = (float*)malloc(sizeof(float)*linnum);

	// keep record of found peaks
	unsigned int *loc_peaks_fnd;
	unsigned int locN_fnd;

	for(i = 0; i < frameNum; ++i)
	{
		//	x = frame .* w;
		for(j = 0; j < framesize; ++j)
		{
			x[j] = frames[j][i] * window[j];	
		}
		// print
		if(0){
			for( j = 0; j < framesize; ++j)
			{
				printf("x[%d]=%3e\n", j+1, x[j]);
			}
		}
		// X = fft(x, NFFT)/framesize;
		FFT_F(x,NFFT,X);

		for(j=0;j<NFFT;++j){
			X[j].real = X[j].real/(float)framesize;
			X[j].imag = X[j].imag/(float)framesize;
		}

		if(0)
		{
			for(j = 0; j < NFFT; ++j)
			{
				printf("%12e + i %12e\n", X[j].real, X[j].imag);
			}
		}

		// f = Fs/2*linspace(0, 1, NFFT/2 + 1);

		// [~, peaklocations] = findpeaks(abs(X(1:(NFFT/2 + 1))), 'SORTSTR', 'descend');
		for(j = 0; j < linnum; ++j)
		{
			absX[j]= sqrt( pow(X[j].real,2.0) + pow(X[j].imag,2.0));
			if(0)	printf("absX[%d] = %5e\n",j+1, absX[j]);
		}

		locN_fnd = 0;
		loc_peaks_fnd = findpeaks(absX,linnum,&locN_fnd);

		if(0)
		{
			for(j=0;j<locN_fnd;j++){
				printf("peak location[%d]=%d\n",j+1,loc_peaks_fnd[j]+1);
			}	
		}

		//---------------------------------------------------------------------------------------
		//	if isempty(peaklocations)
		//		peaklocations = ones(D, 1);
		//	elseif length(peaklocations) < D
		//		peaklocations = padarray(peaklocations, D - length(peaklocations), 1, 'post');
		//	end
		//		framefrequencies(:, i) = f(peaklocations(1:D))';
		//---------------------------------------------------------------------------------------

		if(locN_fnd ==0){
			for(j=0;j<featureDim;++j){
				framefrequencies[j][i] = f[0];
			}
		}else if(locN_fnd < featureDim){
			//pad with 1st element in absX[]
			for(j=0;j<locN_fnd;++j){
				framefrequencies[j][i] = f[loc_peaks_fnd[j]];
			}
			for(j=locN_fnd;j<featureDim;++j){
				framefrequencies[j][i] = f[0];
			}

		}else{
			for(j=0;j<featureDim;++j){
				if(0){
					printf("loc_peaks=%d\n",loc_peaks_fnd[j]);
					printf("f[loc_peaks]=%e\n",f[loc_peaks_fnd[j]]);
				}
				framefrequencies[j][i] = f[loc_peaks_fnd[j]]; //  avoided transpose
			}

		}

/*
		printf("Iter %d:\n",i);
		for(j=0;j<featureDim;++j){
			printf("%.5f ",framefrequencies[j][i]);
		}
		printf("\n");
*/
	}




	// release
	free(x);
	free(f);
	free(absX);
	free(X);

}



