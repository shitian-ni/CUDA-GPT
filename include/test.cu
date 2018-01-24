// //
// //  Created by Shitian Ni on 1/18/18.
// //

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <time.h>
// #include<iostream>
// #include<string.h>
// #include "parameter.h"

// using namespace std;


// __device__ double d_g[G_NUM];
// void*  d_g_ptr;

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

// __device__ void customAdd(double* sdata,double* g_odata, int g_i){
//  	int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int tid = ty * blockDim.x + tx;
// 	// do reduction in shared mem
// 	if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
// 	if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
// 	if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
// 	if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
// 	if (tid < 32){
// 		sdata[tid] += sdata[tid + 32];__syncthreads();
// 		sdata[tid] += sdata[tid + 16];__syncthreads();
// 		sdata[tid] += sdata[tid + 8];__syncthreads();
// 		sdata[tid] += sdata[tid + 4];__syncthreads();
// 		sdata[tid] += sdata[tid + 2];__syncthreads();
// 		sdata[tid] += sdata[tid + 1];__syncthreads();
// 	}

// 	// write result for this block to global mem
// 	if (tid == 0) atomicAdd(&g_odata[g_i]        , sdata[tid]);
// 	// atomicAdd(&g_odata[g_i]        , sdata[tid]);
// }

// __global__ void weightedAVG() {

// 	__shared__ double sdata[TPB_X_TPB];

//     int x1 = blockIdx.x*blockDim.x + threadIdx.x;
//     int y1 = blockIdx.y*blockDim.y + threadIdx.y;
//     int margin = 2;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int tid = ty * blockDim.x + tx;

//     sdata[tid]=1;
//     if ((y1 >= ROW-margin) || (x1 >= COL-margin) || (y1 < margin) || (x1 < margin) ) {
//         sdata[tid]=0;
//     }
//     __syncthreads();
//     customAdd(sdata,d_g,0);
// };

// double g[G_NUM];

// dim3 numBlock;
// dim3 numThread;


// int main (){

// 	gpuErrchk( cudaGetSymbolAddress(&d_g_ptr,d_g));
// 	gpuErrchk( cudaMemset(d_g_ptr, 0, G_NUM * sizeof(double)));

// 	numBlock.x = iDivUp(COL, TPB);
// 	numBlock.y = iDivUp(ROW, TPB);
// 	numThread.x = TPB;
// 	numThread.y = TPB;

// 	weightedAVG<<<numBlock, numThread>>>();


// 	gpuErrchk( cudaDeviceSynchronize() );
//     gpuErrchk( cudaThreadSynchronize() ); // Checks for execution error
// 	gpuErrchk( cudaPeekAtLastError() ); // Checks for launch error
// 	gpuErrchk( cudaMemcpy(g, d_g_ptr, G_NUM*sizeof(double), cudaMemcpyDeviceToHost));
	

// 	cout<<g[0]<<endl;

// 	int ans=(ROW-4)*(COL-4);
// 	cout<<"correct ans: "<<ans<<endl;
// 	return 0;
// }