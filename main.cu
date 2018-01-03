//
//  Created by Shitian Ni on 1/1/18.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<iostream>
#include<string.h>

using namespace std;

#define TPB 32

#define ROW 64
#define COL 64
#define ROW2 64
#define COL2 64
#define CX 32
#define CY 32
#define PI 3.1415926
#define MARGINE 0
#define MAX_BUFFERSIZE 256
#define MAX_IMAGESIZE 1024
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define NoDIRECTION 20
#define G_NUM 27
#define COLH COL*162
#define COLHt COL*27


double H[ROW][COLH], Ht[ROW][COLHt];

void load_image_data( ); /* image input */
void save_image_data( ); /* image output*/
void load_image_file(char *); /* image input */
void save_image_file(char *); /* image output*/

unsigned char image1[1024][1024];
unsigned char image2[1024][1024];

__device__ double d_H[ROW][COLH], d_Ht[ROW][COLHt];
__device__ unsigned char d_image1[1024][1024];
__device__ unsigned char d_image2[1024][1024];
__device__ double d_g[G_NUM], d_g_can1[ROW][COL], d_g_nor1[ROW][COL];
__device__ int d_g_ang1[ROW][COL];

int x_size1 = COL, y_size1 = ROW; /* width & height of image1*/
int x_size2, y_size2; /* width & height of image2 */

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

__global__ void Ht_1() {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW) || (x >= 27 * COL)) {
        return;
    }

    d_Ht[y][x] =  d_H[y][x + COL * 27 * 5];
};
__global__ void Ht_2() {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW) || (x >= 27 * COL)) {
        return;
    }

    d_Ht[y][x] =  d_H[y][x];
};
__global__ void Ht_3(int count, double newVar) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y >= ROW) || (x >= 27 * COL)) {
        return;
    }
    // double varTable[6] = {1.0 / 32, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0};
    double var_p_1 = pow(2.0,count + 1 -5);
    double var = var_p_1 / 2.0;
    d_Ht[y][x] = d_H[y][x + COL * 27 * count] + (d_H[y][x + COL * 27 * (count + 1)] - d_H[y][x + COL * 27 * count]) / (var_p_1 - var) * (newVar - var);
};

#define ROW_X_COL ROW*COL

//1000 times 1200~1300ms
//http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
__device__ void customAdd(double* g_idata,double* g_odata, int g_i){
	__shared__ double sdata[ROW_X_COL];
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
    int x1 = blockIdx.x*blockDim.x + threadIdx.x;
    int y1 = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y1 >= ROW) || (x1 >= COL)) {
        return;
    }
    double dx1, dy1;

    dy1 = y1 - CY;
	dx1 = x1 - CX;

	int thre = (d_g_ang1[y1][x1] + 1) * 3 * COL;

	double t0     = d_Ht[y1][thre + x1]           * d_g_can1[y1][x1];
	double tx2    = d_Ht[y1][thre + x1 + COL]     * d_g_can1[y1][x1];
	double ty2    = d_Ht[y1][thre + x1 + COL * 2] * d_g_can1[y1][x1];

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

#define g0 g[0]
#define gx1 g[1]
#define gy1 g[2]
#define gx1p1 g[3]
#define gx1p2 g[4]
#define gx1p3 g[5]
#define gx1p4 g[6]
#define gy1p1 g[7]
#define gy1p2 g[8]
#define gy1p3 g[9]
#define gy1p4 g[10]
#define gx1p1y1p1 g[11]
#define gx1p2y1p1 g[12]
#define gx1p3y1p1 g[13]
#define gx1p1y1p2 g[14]
#define gx1p2y1p2 g[15]
#define gx1p1y1p3 g[16]
#define gx1x2 g[17]
#define gy1x2 g[18]
#define gx1y2 g[19]
#define gy1y2 g[20]
#define gx2 g[21]
#define gy2 g[22]
#define gx1p2x2 g[23]
#define gx1y1y2 g[24]
#define gx1y1x2 g[25]
#define gy1p2y2 g[26]

#define __1000times 0

int main(){

	dim3 numBlock(iDivUp(COL, TPB), iDivUp(ROW, TPB));
	dim3 numThread(TPB,TPB);

	clock_t begin, end, m_begin, m_end;
	double elapsed_secs=0, m_elapsed_secs=0;
	begin = clock();


	//Get CUDA global memory pointers
	m_begin = clock();
	void* d_image1_ptr; void* d_image2_ptr; void* d_H_ptr;void*  d_Ht_ptr;void*  d_g_ptr;void*  d_g_can1_ptr;void*  d_g_nor1_ptr;void*  d_g_ang1_ptr; 
	cudaGetSymbolAddress(&d_image1_ptr,d_image1);
	// cudaGetSymbolAddress(&d_image2_ptr,d_image2);
	cudaGetSymbolAddress(&d_H_ptr,d_H);
	cudaGetSymbolAddress(&d_Ht_ptr,d_Ht);
	cudaGetSymbolAddress(&d_g_ptr,d_g);
	// cudaGetSymbolAddress(&d_g_can1_ptr,d_g_can1);
	// cudaGetSymbolAddress(&d_g_nor1_ptr,d_g_nor1);
	// cudaGetSymbolAddress(&d_g_ang1_ptr,d_g_ang1);
	m_end = clock();
  	m_elapsed_secs += double(m_end - m_begin) / CLOCKS_PER_SEC * 1000;

	#if __1000times == 1
	for(int tcase=0;tcase<1000;tcase++){
	#endif

	int image3[ROW2][COL2], image4[ROW][COL];					
	int x1, y1, x2, y2, x, y, thre, count;

	// double g0, gx1, gy1, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3;
	// double gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2;

	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double denom;
	double dx1, dx2, dy1, dy2;
	double g_can1[ROW][COL], g_nor1[ROW][COL];
	int g_ang1[ROW][COL];

	double g[G_NUM];
	memset(g,0,sizeof(g));

	double newVar = rand() % 6;

	read_H();

	/* Read image */
	char fileName[128];
	sprintf(fileName, "tests0.pgm"); //
	load_image_file(fileName);

	m_begin = clock();
	cudaMemcpy(d_image1_ptr, image1, 1024 * 1024 *sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemset(d_cuda_defcan_vars, 0, 2 * sizeof(double));
	m_end = clock();
  	m_elapsed_secs += double(m_end - m_begin) / CLOCKS_PER_SEC * 1000;

	cuda_defcan1<<<numBlock, numThread>>>();
	cuda_defcan2<<<numBlock, numThread>>>();
	cuda_roberts8<<<numBlock, numThread>>>();

	m_begin = clock();
	cudaMemcpy(d_H_ptr, H, ROW*162*COL*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(d_g_ptr, 0, G_NUM * sizeof(double));
	m_end = clock();
  	m_elapsed_secs += double(m_end - m_begin) / CLOCKS_PER_SEC * 1000;

  	numBlock.x = iDivUp(27 * COL , TPB);

	if (newVar > 1.0) {
		Ht_1<<<numBlock, numThread>>>();
	} else if (newVar < 1.0 / 32.0) {
		Ht_2<<<numBlock, numThread>>>();
	} else {
		int count = floor(log2(newVar)) + 5;
		Ht_3<<<numBlock, numThread>>>(count, newVar);
	}

	

	weightedAVG<<<numBlock, numThread>>>();
	m_begin = clock();

	cudaMemcpy(g, d_g_ptr, G_NUM*sizeof(double), cudaMemcpyDeviceToHost);
	m_end = clock();
  	m_elapsed_secs += double(m_end - m_begin) / CLOCKS_PER_SEC * 1000;

  	(cudaPeekAtLastError());
	#if __1000times == 1
	}
	#endif
	
	
	end = clock();
  	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;
  	printf("Time elapsed in calculation: %.7f ms\n",elapsed_secs-m_elapsed_secs);
  	printf("Time elapsed in total: %.7f ms\n",elapsed_secs);

  	#if __1000times == 0
	printf("g0 = %f\n", g0);
	#endif

	return 0;
}



void load_image_data( ) {
	/* Input of header & body information of pgm file */
	/* for image1[ ][ ]�警_size1�軽_size1 */
	char file_name[MAX_FILENAME];
	char buffer[MAX_BUFFERSIZE];
	FILE *fp; /* File pointer */
	int max_gray; /* Maximum gray level */
	int x, y; /* Loop variable */

	/* Input file open */
	printf("\n-----------------------------------------------------\n");
	printf("Monochromatic image file input routine \n");
	printf("-----------------------------------------------------\n\n");
	printf("     Only pgm binary file is acceptable\n\n");
	printf("Name of input image file? (*.pgm) : ");
	scanf("%s", file_name);
	fp = fopen(file_name, "rb");
	if (NULL == fp) {
		printf("     The file doesn't exist!\n\n");
		exit(1);
	}
	/* Check of file-type ---P5 */
	fgets(buffer, MAX_BUFFERSIZE, fp);
	if (buffer[0] != 'P' || buffer[1] != '5') {
		printf("     Mistaken file format, not P5!\n\n");
		exit(1);
	}
	/* input of x_size1, y_size1 */
	x_size1 = 0;
	y_size1 = 0;
	while (x_size1 == 0 || y_size1 == 0) {
		fgets(buffer, MAX_BUFFERSIZE, fp);
		if (buffer[0] != '#') {
			sscanf(buffer, "%d %d", &x_size1, &y_size1);
		}
	}
	/* input of max_gray */
	max_gray = 0;
	while (max_gray == 0) {
		fgets(buffer, MAX_BUFFERSIZE, fp);
		if (buffer[0] != '#') {
			sscanf(buffer, "%d", &max_gray);
		}
	}
	/* Display of parameters */
	printf("\n     Image width = %d, Image height = %d\n", x_size1, y_size1);
	printf("     Maximum gray level = %d\n\n",max_gray);
	if (x_size1 > MAX_IMAGESIZE || y_size1 > MAX_IMAGESIZE) {
		printf("     Image size exceeds %d x %d\n\n",
				MAX_IMAGESIZE, MAX_IMAGESIZE);
		printf("     Please use smaller images!\n\n");
		exit(1);
	}
	if (max_gray != MAX_BRIGHTNESS) {
		printf("     Invalid value of maximum gray level!\n\n");
		exit(1);
	}
	/* Input of image data*/
	for (y = 0; y < y_size1; y++) {
		for (x = 0; x < x_size1; x++) {
			image1[y][x] = (unsigned char)fgetc(fp);
		}
	}
	printf("-----Image data input OK-----\n\n");
	printf("-----------------------------------------------------\n\n");
	fclose(fp);
}

void save_image_data( ) {
	/* Output of image2[ ][ ], x_size2, y_size2 in pgm format*/
	char file_name[MAX_FILENAME];
	FILE *fp; /* File pointer */
	int x, y; /* Loop variable */

	/* Output file open */
	printf("-----------------------------------------------------\n");
	printf("Monochromatic image file output routine\n");
	printf("-----------------------------------------------------\n\n");
	printf("Name of output image file? (*.pgm) : ");
	scanf("%s",file_name);
	fp = fopen(file_name, "wb");
	/* output of pgm file header information */
	fputs("P5\n", fp);
	fputs("# Created by Image Processing\n", fp);
	fprintf(fp, "%d %d\n", x_size2, y_size2);
	fprintf(fp, "%d\n", MAX_BRIGHTNESS);
	/* Output of image data */
	for (y = 0; y < y_size2; y++) {
		for (x = 0; x < x_size2; x++) {
			fputc(image2[y][x], fp);
		}
	}
	printf("\n-----Image data output OK-----\n\n");
	printf("-----------------------------------------------------\n\n");
	fclose(fp);
}

void load_image_file(char *filename) {
	/* Input of header & body information of pgm file */
	/* for image1[ ][ ]�警_size1�軽_size1 */
	char buffer[MAX_BUFFERSIZE];
	FILE *fp; /* File pointer */
	int max_gray; /* Maximum gray level */
	int x, y; /* Loop variable */
	/* Input file open */
	fp = fopen(filename, "rb");
	if (NULL == fp) {
		printf("     The file doesn't exist! : %s \n\n", filename);
		exit(1);
	}
	/* Check of file-type ---P5 */
	fgets(buffer, MAX_BUFFERSIZE, fp);
	if (buffer[0] != 'P' || buffer[1] != '5') {
		printf("     Mistaken file format, not P5!\n\n");
		exit(1);
	}
	/* input of x_size1, y_size1 */
	x_size1 = 0;
	y_size1 = 0;
	while (x_size1 == 0 || y_size1 == 0) {
		fgets(buffer, MAX_BUFFERSIZE, fp);
		if (buffer[0] != '#') {
			sscanf(buffer, "%d %d", &x_size1, &y_size1);
		}
	}
	//printf("xsize = %d, ysize = %d", x_size1, y_size1);
	/* input of max_gray */
	max_gray = 0;
	while (max_gray == 0) {
		fgets(buffer, MAX_BUFFERSIZE, fp);
		if (buffer[0] != '#') {
			sscanf(buffer, "%d", &max_gray);
		}
	}
	if (x_size1 > MAX_IMAGESIZE || y_size1 > MAX_IMAGESIZE) {
		printf("     Image size exceeds %d x %d\n\n",
				MAX_IMAGESIZE, MAX_IMAGESIZE);
		printf("     Please use smaller images!\n\n");
		exit(1);
	}
	if (max_gray != MAX_BRIGHTNESS) {
		printf("     Invalid value of maximum gray level!\n\n");
		exit(1);
	}
	/* Input of image data*/
	for (y = 0; y < y_size1; y++) {
		for (x = 0; x < x_size1; x++) {
			image1[y][x] = (unsigned char)fgetc(fp);
		}
	}
	fclose(fp);
}

void save_image_file(char *filename) {
	/* Output of image2[ ][ ], x_size2, y_size2 */
	/* into pgm file with header & body information */
	FILE *fp; /* File pointer */
	int x, y; /* Loop variable */

	fp = fopen(filename, "wb");
	/* output of pgm file header information */
	fputs("P5\n", fp);
	fputs("# Created by Image Processing\n", fp);
	fprintf(fp, "%d %d\n", x_size2, y_size2);
	fprintf(fp, "%d\n", MAX_BRIGHTNESS);
	/* Output of image data */
	for (y = 0; y < y_size2; y++) {
		for (x = 0; x < x_size2; x++) {
			fputc(image2[y][x], fp);
		}
	}
	fclose(fp);
}