// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <time.h>
// #include<iostream>
// #include<string.h>
// #include "parameter.h"

// using namespace std;

// __device__ double d_H[ROW_H][COL_H];

// int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); };

// //https://stackoverflow.com/a/14038590
// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//    if (code != cudaSuccess) 
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }

// void* d_H_ptr;
// dim3 numBlock;
// dim3 numThread;

// int main(){
	
// 	double H[ROW_H][COL_H];

// 	numBlock.x = iDivUp(COL, TPB);
// 	numBlock.y = iDivUp(ROW, TPB);
// 	numThread.x = TPB;
// 	numThread.y = TPB;

// 	cudaGetSymbolAddress(&d_H_ptr,d_H);


// 	cudaDeviceSynchronize();

// 	gpuErrchk( cudaPeekAtLastError() ); // Checks for launch error
//     gpuErrchk( cudaThreadSynchronize() ); // Checks for execution error
// 	gpuErrchk( cudaMemcpy(d_H_ptr, H, ROW_H*COL_H*sizeof(double), cudaMemcpyHostToDevice));
// 	gpuErrchk( cudaDeviceSynchronize() );


// }