//
//  Created by Shitian Ni on 1/18/18.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<iostream>
#include<string.h>
#include "parameter.h"

using namespace std;

__device__ double d_H[ROW_H][COL_H], d_Ht[ROW][COL_Ht];
__device__ unsigned char d_image1[1024][1024];
__device__ unsigned char d_image2[1024][1024];
__device__ double d_g[G_NUM], d_g_can1[ROW][COL], d_g_nor1[ROW][COL];
__device__ int d_g_ang1[ROW][COL];
__device__ char d_sHoG1[ROW - 4][COL - 4];




int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); };

//https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void Ht_1() {
	int margin = 2;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW - 2 * margin) || (x >= 3 * 64 * (COL - 2 * margin))) {
        return;
    }

    d_Ht[y][x] =  d_H[y][x + (COL - 2 * margin) * 3 * 64 * 5];
};
__global__ void Ht_2() {
	int margin = 2;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW - 2 * margin) || (x >= 3 * 64 * (COL - 2 * margin))) {
        return;
    }

    d_Ht[y][x] =  d_H[y][x];   
};
__global__ void Ht_3(int count, double newVar) {
	int margin = 2;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW - 2 * margin) || (x >= 3 * 64 * (COL - 2 * margin))) {
        return;
    }
    double var_p_1 = pow(2.0,count + 1 -5);
    double var = var_p_1 / 2.0;
    d_Ht[y][x] = d_H[y][x + (COL - 2 * margin) * 3 * 64 * count] +
                                   (d_H[y][x + (COL - 2 * margin) * 3 * 64 * (count + 1)] - d_H[y][x + (COL - 2 * margin) * 3 * 64 * count])
                                 / (var_p_1 - var)
                                 * (newVar - var);                    
};


//1000 times 1200~1300ms
//http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
template<typename T>
__device__ void customAdd(T* sdata,T* g_odata){
 	int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
	// do reduction in shared mem
	if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
	if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
	if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
	if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
	if (tid < 32){ sdata[tid] += sdata[tid + 32]; }__syncthreads();
	if (tid < 16){ sdata[tid] += sdata[tid + 16]; }__syncthreads();
	if (tid < 8){ sdata[tid] += sdata[tid + 8]; }__syncthreads();
	if (tid < 4){ sdata[tid] += sdata[tid + 4]; }__syncthreads();
	if (tid < 2){ sdata[tid] += sdata[tid + 2]; }__syncthreads();
	if (tid < 1){ sdata[tid] += sdata[tid + 1]; }__syncthreads();
	// write result for this block to global mem
	if (tid == 0) {atomicAdd(g_odata        , sdata[tid]);}

}
__global__ void weightedAVG() {

	// __shared__ double sdata[TPB_X_TPB];
	__shared__ double sdata[6][TPB_X_TPB];

    int x1 = blockIdx.x*blockDim.x + threadIdx.x;
    int y1 = blockIdx.y*blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int margin = 2;

    double sHoGnumber[64] = sHoGNUMBER;
	double dx1=x1 - CX;
	double dy1=y1 - CY;
	bool condition = ((y1 >= margin) && (x1 >= margin) && (y1 < ROW-margin) && (x1 < COL-margin) && d_sHoG1[y1 - margin][x1 - margin] != -1);
	double t0 = 0;
	double tx2 = 0;
	double ty2 = 0;
	int thre = -1;
    for (int s = 0 ; condition && s < 64 ; s++) {
        if (d_sHoG1[y1 - margin][x1 - margin] == sHoGnumber[s]) {
            thre = s * 3 * (COL - 2 * margin);
            t0     = d_Ht[y1 - margin][thre + x1 - margin]                          * d_g_can1[y1][x1];
		    tx2    = d_Ht[y1 - margin][thre + x1 - margin + (COL - 2 * margin)]     * d_g_can1[y1][x1];
		    ty2    = d_Ht[y1 - margin][thre + x1 - margin + (COL - 2 * margin) * 2] * d_g_can1[y1][x1];
            break;
        }
    }
	

    sdata[0][tid]=t0; 
	sdata[1][tid]=tx2; 
	sdata[2][tid]=ty2; 
	sdata[3][tid]=t0  * dx1;
	sdata[4][tid]=t0  * dx1 * dx1; 
	sdata[5][tid]=t0  * dx1 * dx1 * dx1; 
	__syncthreads(); 

	customAdd(sdata[0],d_g); 
	customAdd(sdata[1],d_g+21);
	customAdd(sdata[2],d_g+22);
	customAdd(sdata[3],d_g+3);
	customAdd(sdata[4],d_g+4);
	customAdd(sdata[5],d_g+5);
	__syncthreads(); 

	sdata[0][tid]=t0  * dx1 * dx1 * dx1 * dx1; 
	sdata[1][tid]=t0  * dy1; 
	sdata[2][tid]=t0  * dy1 * dy1; 
	sdata[3][tid]=t0  * dy1 * dy1 * dy1;  
	sdata[4][tid]=t0  * dy1 * dy1 * dy1 * dy1; 
	sdata[5][tid]=t0  * dx1 * dy1; 
	__syncthreads(); 

	customAdd(sdata[0],d_g+6);
	customAdd(sdata[1],d_g+7);
	customAdd(sdata[2],d_g+8);
	customAdd(sdata[3],d_g+9);
	customAdd(sdata[4],d_g+10); 
	customAdd(sdata[5],d_g+11);
	__syncthreads(); 

	sdata[0][tid]=t0  * dx1 * dx1 * dy1; 
	sdata[1][tid]=t0  * dx1 * dx1 * dx1 * dy1; 
	sdata[2][tid]=t0  * dx1 * dy1 * dy1; 
	sdata[3][tid]=t0  * dx1 * dx1 * dy1 * dy1;
	sdata[4][tid]=t0  * dx1 * dy1 * dy1 * dy1;
	sdata[5][tid]=tx2 * dx1; 
	__syncthreads();  

	customAdd(sdata[0],d_g+12);
	customAdd(sdata[1],d_g+13);
	customAdd(sdata[2],d_g+14);
	customAdd(sdata[3],d_g+15);
	customAdd(sdata[4],d_g+16);
	customAdd(sdata[5],d_g+17);
	__syncthreads();  

	sdata[0][tid]=tx2 * dy1; 
	sdata[1][tid]=ty2 * dx1; 
	sdata[2][tid]=ty2 * dy1;   
	sdata[3][tid]=tx2 * dx1 * dx1; 
	sdata[4][tid]=ty2 * dx1 * dy1;  
	sdata[5][tid]=tx2 * dx1 * dy1; 
	__syncthreads();

	customAdd(sdata[0],d_g+18);
	customAdd(sdata[1],d_g+19);
	customAdd(sdata[2],d_g+20);
	customAdd(sdata[3],d_g+23);
	customAdd(sdata[4],d_g+24);
	customAdd(sdata[5],d_g+25);
	__syncthreads();

	sdata[0][tid]=ty2 * dy1 * dy1; 
	__syncthreads(); 
	customAdd(sdata[0],d_g+26); 
	__syncthreads();  
};

__global__ void cuda_roberts8() {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW) || (x >= COL)) {
        return;
    }

	/* extraction of gradient information by Roberts operator */
	/* with 8-directional codes and strength */
	double delta_RD, delta_LD;
	double angle;

	/* angle & norm of gradient vector calculated
     by Roberts operator */

	if(y >= ROW-1 || x >= COL-1){
		d_g_ang1[y][x] = -1;
		d_g_nor1[y][x] = 0.0;
		return;
	}

	delta_RD = d_image1[y][x + 1] - d_image1[y + 1][x];
	delta_LD = d_image1[y][x]     - d_image1[y + 1][x + 1];
	d_g_nor1[y][x] = sqrt(delta_RD * delta_RD + delta_LD * delta_LD);

	if (d_g_nor1[y][x] == 0.0 || delta_RD * delta_RD + delta_LD * delta_LD < NoDIRECTION * NoDIRECTION) {
		d_g_ang1[y][x] = -1;
		return;
	}
	if (abs(delta_RD) == 0.0) {
		if (delta_LD > 0) d_g_ang1[y][x] = 3;
		else if (delta_LD < 0) d_g_ang1[y][x] = 7;
		else d_g_ang1[y][x] = -1;
		return;
	} 
	angle = atan2(delta_LD, delta_RD);
	if (     angle >  7.0 / 8.0 * PI) d_g_ang1[y][x] = 5;
	else if (angle >  5.0 / 8.0 * PI) d_g_ang1[y][x] = 4;
	else if (angle >  3.0 / 8.0 * PI) d_g_ang1[y][x] = 3;
	else if (angle >  1.0 / 8.0 * PI) d_g_ang1[y][x] = 2;
	else if (angle > -1.0 / 8.0 * PI) d_g_ang1[y][x] = 1;
	else if (angle > -3.0 / 8.0 * PI) d_g_ang1[y][x] = 0;
	else if (angle > -5.0 / 8.0 * PI) d_g_ang1[y][x] = 7;
	else if (angle > -7.0 / 8.0 * PI) d_g_ang1[y][x] = 6;
	else d_g_ang1[y][x] = 5;	
}

/*
	d_cuda_defcan_vars[0]:  mean
	d_cuda_defcan_vars[1]:  norm
	d_cuda_defcan_vars[2]:  npo
*/
__device__ double d_cuda_defcan_vars[3];
__global__ void cuda_defcan1() {
	int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW) || (x >= COL)) {
        return;
    }

	/* definite canonicalization */
	double ratio; // mean: mean value, norm: normal factor, ratio:
	int margine = CANMARGIN / 2;
	int condition = ((x>=margine && y>=margine) && 
					(x<COL-margine)&&(y<ROW-margine) &&
					d_image1[y][x]!=WHITE);
	// if(condition==0)return;
	double this_pixel = condition*(double)d_image1[y][x];
	__shared__ double sdata[3][TPB_X_TPB];
	sdata[0][tid] = this_pixel;
	sdata[1][tid] = this_pixel*this_pixel;
	sdata[2][tid] = condition;

	__syncthreads();

	customAdd(sdata[0],d_cuda_defcan_vars);
	customAdd(sdata[1],d_cuda_defcan_vars+1);
	customAdd(sdata[2],d_cuda_defcan_vars+2);
}
__global__ void cuda_defcan2() {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((y >= ROW) || (x >= COL)) {
        return;
    }

	/*
		s_vars[0]:  mean
		s_vars[1]:  norm
	*/
	__shared__ double s_vars[2];
	if(threadIdx.x == 0 && threadIdx.y == 0){
    	double npo = d_cuda_defcan_vars[2];
		double mean = d_cuda_defcan_vars[0]/ (double)npo;
		double norm = d_cuda_defcan_vars[1] - (double)npo * mean * mean;
		if (norm == 0.0) norm = 1.0;
		s_vars[0] = mean;
		s_vars[1] = norm;
	}
	__syncthreads();

	int condition = ((x<COL-CANMARGIN)&&(y<ROW-CANMARGIN) &&
					d_image1[y][x]!=WHITE);
	// if(condition==0)return;
	double ratio = 1.0 / sqrt(s_vars[1]);
	d_g_can1[y][x] = ratio * ((double)d_image1[y][x] - s_vars[0]);
}

void* d_image1_ptr; void* d_image2_ptr; void* d_H_ptr;void*  d_Ht_ptr;void*  d_g_ptr;
void*  d_g_can1_ptr;void*  d_g_nor1_ptr;void*  d_g_ang1_ptr;void* d_sHoG1_ptr;
void* d_cuda_defcan_vars_ptr;
double g[G_NUM];

dim3 numBlock;
dim3 numThread;

void cuda_init_parameter(){
	numBlock.x = iDivUp(COL, TPB);
	numBlock.y = iDivUp(ROW, TPB);
	numThread.x = TPB;
	numThread.y = TPB;

	gpuErrchk( cudaGetSymbolAddress(&d_image1_ptr,d_image1));
	gpuErrchk( cudaGetSymbolAddress(&d_H_ptr,d_H));
	gpuErrchk( cudaGetSymbolAddress(&d_Ht_ptr,d_Ht));
	gpuErrchk( cudaGetSymbolAddress(&d_g_ptr,d_g));
	gpuErrchk( cudaGetSymbolAddress(&d_sHoG1_ptr,d_sHoG1));
	gpuErrchk( cudaGetSymbolAddress(&d_g_can1_ptr,d_g_can1));
	gpuErrchk( cudaGetSymbolAddress(&d_g_nor1_ptr,d_g_nor1));
	gpuErrchk( cudaGetSymbolAddress(&d_g_ang1_ptr,d_g_ang1));
	gpuErrchk( cudaGetSymbolAddress(&d_cuda_defcan_vars_ptr,d_cuda_defcan_vars));

	
	gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaThreadSynchronize() ); // Checks for execution error
	gpuErrchk( cudaPeekAtLastError() ); // Checks for launch error
}

__global__ void test(){
	// int x = blockIdx.x*blockDim.x + threadIdx.x;
 //    int y = blockIdx.y*blockDim.y + threadIdx.y;
 //    if ((y >= ROW - 2 * margin) || (x >= 3 * 64 * (COL - 2 * margin))) {
 //        return;
 //    }
}

void cuda_procImg(double g_can[ROW][COL], int g_ang[ROW][COL], double g_nor[ROW][COL], char g_HoG[ROW][COL][8], char sHoG[ROW - 4][COL - 4], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]){
	cudaMemset(d_cuda_defcan_vars_ptr, 0, 3 * sizeof(double));
	cudaMemcpy(d_image1_ptr, image1, MAX_IMAGESIZE*MAX_IMAGESIZE*sizeof(unsigned char), cudaMemcpyHostToDevice);
	numBlock.x = iDivUp(COL, TPB);
	numBlock.y = iDivUp(ROW, TPB);
	cuda_defcan1<<<numBlock, numThread>>>();
	cuda_defcan2<<<numBlock, numThread>>>();
	cuda_roberts8<<<numBlock, numThread>>>();
	cudaMemcpy(g_can, d_g_can1_ptr, ROW*COL*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(g_ang, d_g_ang1_ptr, ROW*COL*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(g_nor, d_g_nor1_ptr, ROW*COL*sizeof(double), cudaMemcpyDeviceToHost);
}
void cuda_calc_defcan1(double g_can1[ROW][COL], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]){
	cudaMemset(d_cuda_defcan_vars_ptr, 0, 3 * sizeof(double));
	cudaMemcpy(d_image1_ptr, image1, MAX_IMAGESIZE*MAX_IMAGESIZE*sizeof(unsigned char), cudaMemcpyHostToDevice);
	numBlock.x = iDivUp(COL, TPB);
	numBlock.y = iDivUp(ROW, TPB);
	cuda_defcan1<<<numBlock, numThread>>>();
	cuda_defcan2<<<numBlock, numThread>>>();
	cuda_roberts8<<<numBlock, numThread>>>();
	cudaMemcpy(g_can1, d_g_can1_ptr, ROW*COL*sizeof(double), cudaMemcpyDeviceToHost);
}

void cuda_update_parameter(int g_ang1[ROW][COL], double g_can1[ROW][COL],double H[ROW_H][COL_H],char sHoG1[ROW - 4][COL - 4]){

	// cudaMemcpy(d_g_ang1_ptr, g_ang1, ROW*COL*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sHoG1_ptr, sHoG1, (ROW - 4)*(COL-4)*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_H_ptr, H, ROW_H*COL_H*sizeof(double), cudaMemcpyHostToDevice);
}

void cuda_Ht(double newVar){
	int margin = 2;
	numBlock.x = iDivUp(3 * 64 * (COL - 2 * margin), TPB);
	numBlock.y = iDivUp(ROW - 2 * margin, TPB);
	if (newVar > 1.0) {
		Ht_1<<<numBlock, numThread>>>();
	} else if (newVar < 1.0 / 32.0) {
		Ht_2<<<numBlock, numThread>>>();
	} else {
		int count = floor(log2(newVar)) + 5;
		Ht_3<<<numBlock, numThread>>>(count, newVar);
	}
	// gpuErrchk( cudaDeviceSynchronize() );
 //    gpuErrchk( cudaThreadSynchronize() ); // Checks for execution error
	// gpuErrchk( cudaPeekAtLastError() ); // Checks for launch error
}
double* cuda_calc_g(){
	cudaMemset(d_g_ptr, 0, G_NUM * sizeof(double));
	numBlock.x = iDivUp(COL, TPB);
	numBlock.y = iDivUp(ROW, TPB);
	weightedAVG<<<numBlock, numThread>>>();
	// gpuErrchk( cudaPeekAtLastError() );
 //    gpuErrchk( cudaThreadSynchronize() ); // Checks for execution error
	// gpuErrchk( cudaDeviceSynchronize() );
	cudaMemcpy(g, d_g_ptr, G_NUM*sizeof(double), cudaMemcpyDeviceToHost);
	return g;
}