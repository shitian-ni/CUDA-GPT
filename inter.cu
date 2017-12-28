#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<iostream>

using namespace std;

#define ROW 64
#define COL 64


#define TPB 32

double varTable[6] = {1.0 / 32, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0};
double H[ROW][COL * 162], Ht[ROW][COL * 27];

void read_H(){
	char fileName[128];
	sprintf(fileName, "tests0_temp");
	FILE *fp;
	if((fp = fopen(fileName, "rb")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	fread(H, sizeof(double), 162 * COL * ROW, fp);
}


int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); };

__global__ void Ht_1(double *d_Ht, double *d_H) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW) || (x >= 27 * COL)) {
        return;
    }
    d_Ht[y * 27 * COL+x] =  d_H[y* 162 * COL+x + COL * 27 * 5];
};
__global__ void Ht_2(double *d_Ht, double *d_H) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW) || (x >= 27 * COL)) {
        return;
    }
    printf("%d %d %.5f\n",x,y,d_H[y* 162*COL+x]);
    d_Ht[y*27* COL+x] =  d_H[y* 162*COL+x];
};
__global__ void Ht_3(double *d_Ht, double *d_H, double* d_varTable, int count, double newVar) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW) || (x >= 27 * COL)) {
        return;
    }
    d_Ht[y*27 * COL + x] = d_H[y* 162*COL+x + COL * 27 * count] + (d_H[y* 162*COL+x + COL * 27 * (count + 1)] - d_H[y * 162*COL+x + COL * 27 * count]) / (d_varTable[count + 1] - d_varTable[count]) * (newVar - d_varTable[count]);
};

int main(){
	double newVar = 1.0/3;
	
	read_H();

	

	double* d_H;
	double* d_Ht;
	double* d_varTable;

	cudaMalloc(&d_H, ROW*162*COL*sizeof(double));
	cudaMalloc(&d_Ht, ROW*27*COL*sizeof(double));
	cudaMalloc(&d_varTable, 6*sizeof(double));

	cudaMemcpy(d_H, H, ROW*162*COL*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_varTable, varTable, 6*sizeof(double), cudaMemcpyHostToDevice);

	dim3 numBlock(iDivUp(27 * COL , TPB), iDivUp(ROW, TPB));
	dim3 numThread(TPB,TPB);

	if (newVar > 1.0) {
		Ht_1<<<numBlock, numThread>>>(d_Ht, d_H);
	} else if (newVar < 1.0 / 32.0) {
		Ht_2<<<numBlock, numThread>>>(d_Ht, d_H);
	} else {
		int count = floor(log2(newVar)) + 5;
		Ht_3<<<numBlock, numThread>>>(d_Ht, d_H, d_varTable, count, newVar);
	}
	(cudaPeekAtLastError());
    // (cudaDeviceSynchronize());
	cudaMemcpy(Ht, d_Ht, ROW*27*COL*sizeof(double), cudaMemcpyDeviceToHost);

    // (cudaDeviceSynchronize());

	freopen("cuda_Ht.txt","w",stdout);
	for(int i=0;i<ROW;i++){
		for(int j=0;j<27*COL;j++){
			cout<<Ht[i][j]<<" ";
		}
		cout<<endl;
	}
	return 0;
}