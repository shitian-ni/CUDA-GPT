#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<iostream>
#include<map>
#include<string.h>

using namespace std;

#define ROW 64
#define COL 64


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

double varTable[6] = {1.0 / 32, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0};
double H[ROW][COL * 162], Ht[ROW][COL * 27];

void load_image_data( ); /* image input */
void save_image_data( ); /* image output*/
void load_image_file(char *); /* image input */
void save_image_file(char *); /* image output*/
void defcan(double g_can[ROW][COL]);
void roberts8(int g_ang[ROW][COL], double g_nor[ROW][COL]);

unsigned char image1[1024][1024];
unsigned char image2[1024][1024];
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

__global__ void weightedAVG(double *d_Ht, double* d_g_can1, int* d_g_ang1, double* d_g) {
    int x1 = blockIdx.x*blockDim.x + threadIdx.x;
    int y1 = blockIdx.y*blockDim.y + threadIdx.y;
    if ((y1 >= ROW) || (x1 >= COL)) {
        return;
    }
    double dx1, dy1;

    dy1 = y1 - CY;
	dx1 = x1 - CX;

	int thre = (d_g_ang1[y1*COL + x1] + 1) * 3 * COL;

	double t0     = d_Ht[y1*COLHt + thre + x1]           * d_g_can1[y1*COL + x1];
	double tx2    = d_Ht[y1*COLHt + thre + x1 + COL]     * d_g_can1[y1*COL + x1];
	double ty2    = d_Ht[y1*COLHt + thre + x1 + COL * 2] * d_g_can1[y1*COL + x1];

	atomicAdd(&d_g[0]        , t0);
	atomicAdd(&d_g[21]       , tx2);
	atomicAdd(&d_g[22]       , ty2);
	atomicAdd(&d_g[3]     , t0  * dx1);
	atomicAdd(&d_g[4]     , t0  * dx1 * dx1);
	atomicAdd(&d_g[5]     , t0  * dx1 * dx1 * dx1);
	atomicAdd(&d_g[6]     , t0  * dx1 * dx1 * dx1 * dx1);
	atomicAdd(&d_g[7]     , t0  * dy1);
	atomicAdd(&d_g[8]     , t0  * dy1 * dy1);
	atomicAdd(&d_g[9]     , t0  * dy1 * dy1 * dy1);
	atomicAdd(&d_g[10]     , t0  * dy1 * dy1 * dy1 * dy1);
	atomicAdd(&d_g[11] , t0  * dx1 * dy1);
	atomicAdd(&d_g[12] , t0  * dx1 * dx1 * dy1);
	atomicAdd(&d_g[13] , t0  * dx1 * dx1 * dx1 * dy1);
	atomicAdd(&d_g[14] , t0  * dx1 * dy1 * dy1);
	atomicAdd(&d_g[15] , t0  * dx1 * dx1 * dy1 * dy1);
	atomicAdd(&d_g[16] , t0  * dx1 * dy1 * dy1 * dy1);
	atomicAdd(&d_g[17]     , tx2 * dx1);
	atomicAdd(&d_g[18]     , tx2 * dy1);
	atomicAdd(&d_g[19]     , ty2 * dx1);
	atomicAdd(&d_g[20]     , ty2 * dy1);
	atomicAdd(&d_g[23]   , tx2 * dx1 * dx1);
	atomicAdd(&d_g[24]   , ty2 * dx1 * dy1);
	atomicAdd(&d_g[25]   , tx2 * dx1 * dy1);
	atomicAdd(&d_g[26]   , ty2 * dy1 * dy1);
};

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

int main(){
	clock_t begin, end, m_begin, m_end;
	double elapsed_secs=0, m_elapsed_secs=0;
	begin = clock();
	
	int image3[ROW2][COL2], image4[ROW][COL];					
	int x1, y1, x2, y2, x, y, thre, count;

	// double g0, gx1, gy1, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3;
	// double gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2;


	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double denom;
	double dx1, dx2, dy1, dy2;
	double g_can1[ROW][COL], g_nor1[ROW][COL];
	int g_ang1[ROW][COL];
	double newVar = rand() % 6;
	
	read_H();

	/* Read image */
	char fileName[128];
	sprintf(fileName, "tests0.pgm"); //
	load_image_file(fileName);
	defcan(g_can1);
	roberts8(g_ang1, g_nor1);

	double* d_H;
	double* d_Ht;
	double* d_varTable;

	m_begin = clock();

	cudaMalloc(&d_H, ROW*162*COL*sizeof(double));
	cudaMalloc(&d_Ht, ROW*27*COL*sizeof(double));
	cudaMalloc(&d_varTable, 6*sizeof(double));

	cudaMemcpy(d_H, H, ROW*162*COL*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_varTable, varTable, 6*sizeof(double), cudaMemcpyHostToDevice);

	m_end = clock();
  	m_elapsed_secs += double(m_end - m_begin) / CLOCKS_PER_SEC * 1000;

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

	double g[G_NUM];
	memset(g,0,sizeof(g));
	double* d_g;
	int* d_g_ang1;
	double* d_g_can1;

	m_begin = clock();

	cudaMalloc(&d_g, G_NUM*sizeof(double));
	cudaMalloc(&d_g_ang1, ROW * COL * sizeof(int));
	cudaMalloc(&d_g_can1, ROW * COL * sizeof(double));
	cudaMemset(d_g, 0, G_NUM * sizeof(double));
	cudaMemcpy(d_g_ang1, g_ang1, ROW * COL *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_g_can1, g_can1, ROW * COL *sizeof(double), cudaMemcpyHostToDevice);

	m_end = clock();
  	m_elapsed_secs += double(m_end - m_begin) / CLOCKS_PER_SEC * 1000;

	weightedAVG<<<numBlock, numThread>>>(d_Ht, d_g_can1, d_g_ang1, d_g);

	cudaMemcpy(g, d_g, G_NUM*sizeof(double), cudaMemcpyDeviceToHost);
	
	end = clock();
  	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;
  	printf("Time elapsed in calculation: %.7f ms\n",elapsed_secs-m_elapsed_secs);
  	printf("Time elapsed in total: %.7f ms\n",elapsed_secs);

	printf("g0 = %f\n", g0);
	return 0;
}

void roberts8(int g_ang[ROW][COL], double g_nor[ROW][COL]) {
	/* extraction of gradient information by Roberts operator */
	/* with 8-directional codes and strength */
	double delta_RD, delta_LD;
	double angle;
	int x, y;  /* Loop variable */

	/* angle & norm of gradient vector calculated
     by Roberts operator */
	for (y = 0; y < ROW; y++) {
		for (x = 0; x < COL; x++) {
			g_ang[y][x] = -1;
			g_nor[y][x] = 0.0;
		}
	}

	for (y = 0; y < ROW - 1; y++) {
		for (x = 0; x < COL - 1; x++) {
			delta_RD = image1[y][x + 1] - image1[y + 1][x];
			delta_LD = image1[y][x]     - image1[y + 1][x + 1];
			g_nor[y][x] = sqrt(delta_RD * delta_RD + delta_LD * delta_LD);

			if (g_nor[y][x] == 0.0 || delta_RD * delta_RD + delta_LD * delta_LD < NoDIRECTION * NoDIRECTION) continue;

			if (abs(delta_RD) == 0.0) {
				if (delta_LD > 0) g_ang[y][x] = 3;
				if (delta_LD < 0) g_ang[y][x] = 7;
			} else {
				angle = atan2(delta_LD, delta_RD);
				if (     angle >  7.0 / 8.0 * PI) g_ang[y][x] = 5;
				else if (angle >  5.0 / 8.0 * PI) g_ang[y][x] = 4;
				else if (angle >  3.0 / 8.0 * PI) g_ang[y][x] = 3;
				else if (angle >  1.0 / 8.0 * PI) g_ang[y][x] = 2;
				else if (angle > -1.0 / 8.0 * PI) g_ang[y][x] = 1;
				else if (angle > -3.0 / 8.0 * PI) g_ang[y][x] = 0;
				else if (angle > -5.0 / 8.0 * PI) g_ang[y][x] = 7;
				else if (angle > -7.0 / 8.0 * PI) g_ang[y][x] = 6;
				else g_ang[y][x] = 5;
			}
			//printf("(%d, %d) ang = %d,  norm = %f\n", x, y, g_ang[y][x], g_nor[y][x]);
		}
	}
}

void defcan(double g_can[ROW][COL]) {
	/* definite canonicalization */
	int x, y;
	double mean, norm, ratio; // mean: mean value, norm: normal factor, ratio:
	int npo; // number of point

	npo = (ROW - 2 * MARGINE) * (COL - 2 * MARGINE);
	mean = norm = 0.0;
	for (y = MARGINE ; y < ROW - MARGINE ; y++) {
		for (x = MARGINE ; x < COL - MARGINE ; x++) {
			mean += (double)image1[y][x];
			norm += (double)image1[y][x] * (double)image1[y][x];
		}
	}
	mean /= (double)npo;
	norm -= (double)npo * mean * mean;
	if (norm == 0.0) norm = 1.0;
	ratio = 1.0 / sqrt(norm);
	for (y = 0 ; y < ROW; y++) {
		for (x = 0 ; x < COL; x++) {
			g_can[y][x] = ratio * ((double)image1[y][x] - mean);
		}
	}
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