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
    // double varTable[6] = {1.0 / 32, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0};
    double var_p_1 = pow(2.0,count + 1 -5);
    double var = var_p_1 / 2.0;
    d_Ht[y][x] = d_H[y][x + (COL - 2 * margin) * 3 * 64 * count] +
                                   (d_H[y][x + (COL - 2 * margin) * 3 * 64 * (count + 1)] - d_H[y][x + (COL - 2 * margin) * 3 * 64 * count])
                                 / (var_p_1 - var)
                                 * (newVar - var);;
};


//1000 times 1200~1300ms
//http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
__device__ void customAdd(double* g_idata,double* g_odata, int g_i){
	__shared__ double sdata[TPB_X_TPB];
	int x1 = blockIdx.x*blockDim.x + threadIdx.x;
    int y1 = blockIdx.y*blockDim.y + threadIdx.y;
    int id = y1*COL+x1;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    sdata[tid] = g_idata[id];
    __syncthreads();
 
	// do reduction in shared mem
	if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
	if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
	if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
	if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
	if (tid < 32){
		sdata[tid] += sdata[tid + 32];__syncthreads();
		sdata[tid] += sdata[tid + 16];__syncthreads();
		sdata[tid] += sdata[tid + 8];__syncthreads();
		sdata[tid] += sdata[tid + 4];__syncthreads();
		sdata[tid] += sdata[tid + 2];__syncthreads();
		sdata[tid] += sdata[tid + 1];__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) atomicAdd(&g_odata[g_i]        , sdata[tid]);
}

__device__ double d_weightedAVG_data_to_sum[G_NUM][ROW_X_COL];
__global__ void weightedAVG() {
    int margin = 2;

    int x1 = blockIdx.x*blockDim.x + threadIdx.x;
    int y1 = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y1 >= ROW-margin) || (x1 >= COL-margin) || (y1 < margin) || (x1 < margin) ) {
        return;
    }
    double sHoGnumber[64] = sHoGNUMBER;
    double dx1, dy1;

    dy1 = y1 - CY;
	dx1 = x1 - CX;

	if (d_sHoG1[y1 - margin][x1 - margin] == -1) return;

	int thre = -1;

    for (int s = 0 ; s < 64 ; s++) {
        if (d_sHoG1[y1 - margin][x1 - margin] == sHoGnumber[s]) {
            thre = s * 3 * (COL - 2 * margin);
            break;
        }
    }
    printf("d_sHoG1[%d][%d]: %.5f\n",y1-margin,x1-margin,d_sHoG1[y1 - margin][x1 - margin]);
    printf("d_g_can1[%d][%d]: %.5f\n",y1,x1,d_g_can1[y1][x1]);
    // printf("d_sHoG1[%d][%d]: %.5f\n",y1-margin,x1-margin,d_sHoG1[y1 - margin][x1 - margin]);
    if (thre == -1) {
        printf("ERROR! \n");
    }

	double t0     = d_Ht[y1 - margin][thre + x1 - margin]                          * d_g_can1[y1][x1];
    double tx2    = d_Ht[y1 - margin][thre + x1 - margin + (COL - 2 * margin)]     * d_g_can1[y1][x1];
    double ty2    = d_Ht[y1 - margin][thre + x1 - margin + (COL - 2 * margin) * 2] * d_g_can1[y1][x1];

	d_weightedAVG_data_to_sum[0][y1*COL+x1]=t0;
	d_weightedAVG_data_to_sum[21][y1*COL+x1]=tx2;
	d_weightedAVG_data_to_sum[22][y1*COL+x1]=ty2;
	d_weightedAVG_data_to_sum[3][y1*COL+x1]=t0  * dx1;
	d_weightedAVG_data_to_sum[4][y1*COL+x1]=t0  * dx1 * dx1;
	d_weightedAVG_data_to_sum[5][y1*COL+x1]=t0  * dx1 * dx1 * dx1;
	d_weightedAVG_data_to_sum[6][y1*COL+x1]=t0  * dx1 * dx1 * dx1 * dx1;
	d_weightedAVG_data_to_sum[7][y1*COL+x1]=t0  * dy1; 
	d_weightedAVG_data_to_sum[8][y1*COL+x1]=t0  * dy1 * dy1; 
	d_weightedAVG_data_to_sum[9][y1*COL+x1]=t0  * dy1 * dy1 * dy1; 
	d_weightedAVG_data_to_sum[10][y1*COL+x1]=t0  * dy1 * dy1 * dy1 * dy1; 
	d_weightedAVG_data_to_sum[11][y1*COL+x1]=t0  * dx1 * dy1; 
	d_weightedAVG_data_to_sum[12][y1*COL+x1]=t0  * dx1 * dx1 * dy1; 
	d_weightedAVG_data_to_sum[13][y1*COL+x1]=t0  * dx1 * dx1 * dx1 * dy1; 
	d_weightedAVG_data_to_sum[14][y1*COL+x1]=t0  * dx1 * dy1 * dy1; 
	d_weightedAVG_data_to_sum[15][y1*COL+x1]=t0  * dx1 * dx1 * dy1 * dy1; 
	d_weightedAVG_data_to_sum[16][y1*COL+x1]=t0  * dx1 * dy1 * dy1 * dy1; 
	d_weightedAVG_data_to_sum[17][y1*COL+x1]=tx2 * dx1; 
	d_weightedAVG_data_to_sum[18][y1*COL+x1]=tx2 * dy1;
	d_weightedAVG_data_to_sum[19][y1*COL+x1]=ty2 * dx1; 
	d_weightedAVG_data_to_sum[20][y1*COL+x1]=ty2 * dy1; 
	d_weightedAVG_data_to_sum[23][y1*COL+x1]=tx2 * dx1 * dx1; 
	d_weightedAVG_data_to_sum[24][y1*COL+x1]=ty2 * dx1 * dy1; 
	d_weightedAVG_data_to_sum[25][y1*COL+x1]=tx2 * dx1 * dy1; 
	d_weightedAVG_data_to_sum[26][y1*COL+x1]=ty2 * dy1 * dy1;  

	__syncthreads();

	for(int i=0;i<G_NUM;i++){
		customAdd(d_weightedAVG_data_to_sum[i],d_g,i);
	}
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

	// d_g_ang1[y][x] = -1;
	// d_g_nor1[y][x] = 0.0;

	// __syncthreads();

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
*/
__device__ double d_cuda_defcan_vars[2];
__device__ double d_cuda_defcan_to_sum[2][ROW_X_COL];
__global__ void cuda_defcan1() {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW) || (x >= COL)) {
        return;
    }

	/* definite canonicalization */
	double ratio; // mean: mean value, norm: normal factor, ratio:
	int npo; // number of point
	npo = (ROW - 2 * MARGINE) * (COL - 2 * MARGINE);
	
	double this_pixel = (double)d_image1[y][x];
	d_cuda_defcan_to_sum[0][y*COL+x]=this_pixel;
	d_cuda_defcan_to_sum[1][y*COL+x]=this_pixel * this_pixel;

	__syncthreads();

	for(int i=0;i<2;i++){
		customAdd(d_cuda_defcan_to_sum[i],d_cuda_defcan_vars,i);
	}
}
__global__ void cuda_defcan2() {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((y >= ROW) || (x >= COL)) {
        return;
    }

    int npo; // number of point
	npo = (ROW - 2 * MARGINE) * (COL - 2 * MARGINE);
	/*
		s_vars[0]:  mean
		s_vars[1]:  norm
	*/
	__shared__ double s_vars[2];
	if(threadIdx.x == 0 && threadIdx.y == 0){
		double mean = d_cuda_defcan_vars[0]/ (double)npo;
		double norm = d_cuda_defcan_vars[1] - (double)npo * mean * mean;
		if (norm == 0.0) norm = 1.0;
		s_vars[0] = mean;
		s_vars[1] = norm;
	}
	__syncthreads();

	double ratio = 1.0 / sqrt(s_vars[1]);
	d_g_can1[y][x] = ratio * ((double)d_image1[y][x] - s_vars[0]);
}

void* d_image1_ptr; void* d_image2_ptr; void* d_H_ptr;void*  d_Ht_ptr;void*  d_g_ptr;void*  d_g_can1_ptr;void*  d_g_nor1_ptr;void*  d_g_ang1_ptr;void* d_sHoG1_ptr;
double g[G_NUM];

dim3 numBlock;
dim3 numThread;

void cuda_init_parameter(unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]){
	numBlock.x = iDivUp(COL, TPB);
	numBlock.y = iDivUp(ROW, TPB);
	numThread.x = TPB;
	numThread.y = TPB;

	cudaGetSymbolAddress(&d_image1_ptr,d_image1);
	cudaGetSymbolAddress(&d_H_ptr,d_H);
	cudaGetSymbolAddress(&d_Ht_ptr,d_Ht);
	cudaGetSymbolAddress(&d_g_ptr,d_g);
	cudaGetSymbolAddress(&d_sHoG1_ptr,d_sHoG1);
	cudaGetSymbolAddress(&d_g_can1_ptr,d_g_can1);
	cudaGetSymbolAddress(&d_g_nor1_ptr,d_g_nor1);
	cudaGetSymbolAddress(&d_g_ang1_ptr,d_g_ang1);
	

	cudaMemcpy(d_image1_ptr, image1, 1024 * 1024 *sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemset(d_cuda_defcan_vars, 0, 2 * sizeof(double));

	cudaDeviceSynchronize();
}

__global__ void test(){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((y >= 2) || (x >= 2)) {
        return;
    }

    printf("d_g_can1[%d][%d]: %.5f\n",y,x,d_g_can1[y][x]);
}

void cuda_update_parameter(int g_ang1[ROW][COL], double g_can1[ROW][COL],double H[ROW_H][COL_H],char sHoG1[ROW - 4][COL - 4]){
	cudaMemcpy(d_H_ptr, H, ROW_H*COL_H*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_g_ang1_ptr, g_ang1, ROW*COL*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_g_can1_ptr, g_can1, ROW*COL*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sHoG1_ptr, sHoG1, (ROW - 4)*(COL-4)*sizeof(char), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	printf("a");
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			printf("g_can1[%d][%d]: %.5f\n",i,j,g_can1[i][j]);
		}
	}
	numBlock.x = iDivUp(COL, TPB);
	numBlock.y = iDivUp(ROW, TPB);
	printf("b");
	test<<<numBlock, numThread>>>();
	cudaDeviceSynchronize();
	printf("c");
	// double newVar = 0.5;
	// if (newVar > 1.0) {
	// 	Ht_1<<<numBlock, numThread>>>();
	// } else if (newVar < 1.0 / 32.0) {
	// 	Ht_2<<<numBlock, numThread>>>();
	// } else {
	// 	int count = floor(log2(newVar)) + 5;
	// 	Ht_3<<<numBlock, numThread>>>(count, newVar);
	// }
	// int margin = 2;
	// numBlock.x = iDivUp(3 * 64 * (COL - 2 * margin), TPB);
	// numBlock.y = iDivUp(ROW - 2 * margin, TPB);
	// weightedAVG<<<numBlock, numThread>>>();
	// cudaMemcpy(g, d_g_ptr, G_NUM*sizeof(double), cudaMemcpyDeviceToHost);
	// printf("g0 in cuda_calc_g : %.5f\n",g[0]);
	//return g;
}

void cuda_Ht(double newVar){
	// numBlock.x = iDivUp(COL, TPB);
	// numBlock.y = iDivUp(ROW, TPB);
	// if (newVar > 1.0) {
	// 	Ht_1<<<numBlock, numThread>>>();
	// } else if (newVar < 1.0 / 32.0) {
	// 	Ht_2<<<numBlock, numThread>>>();
	// } else {
	// 	int count = floor(log2(newVar)) + 5;
	// 	Ht_3<<<numBlock, numThread>>>(count, newVar);
	// }
	// int margin = 2;
	// numBlock.x = iDivUp(3 * 64 * (COL - 2 * margin), TPB);
	// numBlock.y = iDivUp(ROW - 2 * margin, TPB);
	// weightedAVG<<<numBlock, numThread>>>();
	// cudaMemcpy(g, d_g_ptr, G_NUM*sizeof(double), cudaMemcpyDeviceToHost);
	// printf("g0 in cuda_calc_g : %.5f\n",g[0]);
	// return g;
}
double* cuda_calc_g(){
	// int margin = 2;
	// numBlock.x = iDivUp(3 * 64 * (COL - 2 * margin), TPB);
	// numBlock.y = iDivUp(ROW - 2 * margin, TPB);
	// weightedAVG<<<numBlock, numThread>>>();
	// cudaMemcpy(g, d_g_ptr, G_NUM*sizeof(double), cudaMemcpyDeviceToHost);
	// printf("g0 in cuda_calc_g : %.5f\n",g[0]);
	return g;
}


