#include<iostream>
#include <stdlib.h> 
#include <ctime>
#include<stdio.h>

using namespace std;

#define 



#define NX_F 170 * 2
#define NY_F 136 * 2

#define NX_G 170 * 2
#define NY_G 136*8 * 2




#define TPB 32

int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); };

__global__ void calc_h(int *d_f, int *d_g, int *d_h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= NY_F) || (x >= NX_F)) {
        return;
    }
    if(d_f[y*NX_F+x]<0){
    	return;
    }
    int g_x = d_f[y*NX_F+x]*10+x;
    int g_y = y;

    atomicAdd(d_h, d_g[g_y*NX_G+g_x]);
};


int main(){
	cout<<"NY_G: "<<NY_G<<endl;
	srand(1);
	int f[NY_F][NX_F];
	int g[NY_G][NX_G];
	int h =0 ;
	int* p_h = &h;
	for(int i=0;i<NY_F;i++){
		for(int j=0;j<NX_F;j++){
			f[i][j]=rand()%9-1;
		}
	}
	for(int i=0;i<NY_G;i++){
		for(int j=0;j<NX_G;j++){
			g[i][j]=rand()%19-1;
		}
	}

	clock_t begin = clock();

	int* d_f;
	int* d_g;
	int* d_h;
	cudaMalloc(&d_h, sizeof(int));
	cudaMalloc(&d_f, NY_F*NX_F*sizeof(int));
	cudaMalloc(&d_g, NY_G*NX_G*sizeof(int));
	cudaMemset(d_h, 0, sizeof(int));
	cudaMemcpy(d_f, f, NY_F*NX_F*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_g, g, NY_G*NX_G*sizeof(int), cudaMemcpyHostToDevice);
	dim3 numBlock(iDivUp(NX_F, TPB), iDivUp(NY_F, TPB));
	cout<<iDivUp(NX_F, TPB)<<" "<<iDivUp(NY_F, TPB)<<endl;
	dim3 numThread(TPB,TPB);
	// unsigned int sharedSize = numThreads*sizeof(int);

	// clock_t end = clock();
 //  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;
 //  	printf("Time elapsed for memory: %.7f ms\n",elapsed_secs);

	// begin = clock();
    calc_h<<<numBlock, numThread>>>(d_f, d_g, d_h);

    cudaMemcpy(p_h, d_h, sizeof(int), cudaMemcpyDeviceToHost);
    (cudaPeekAtLastError());
    (cudaDeviceSynchronize());

	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;
  	printf("Time elapsed in total: %.7f ms\n",elapsed_secs);
  	printf("h: %d\n",h);

  	cudaFree(d_f);
  	cudaFree(d_h);
  	cudaFree(d_g);
	return 0;
}