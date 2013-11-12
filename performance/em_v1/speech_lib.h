#ifndef SPEECH_LIB_H
#define SPEECH_LIB_H


void normalise_1d_f(float *x, int len);

void normalise_2d_f(float *x, int row, int col);

void mk_stochastic_2d_f(float *x, int row, int col, float *out);

void symmetrize_f(float *in, int N);

#endif
