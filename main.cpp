//
//  Created by Shizhi Zhang on 1/1/18.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<iostream>

using namespace std;

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

double var[6] = {1.0 / 32, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0};
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

#define __1000times 0

int main(){
	clock_t begin = clock();

	#if __1000times == 1
	for(int tcase=0;tcase<1000;tcase++){
	#endif

	int image3[ROW2][COL2], image4[ROW][COL];					
	int x1, y1, x2, y2, x, y, thre, count;
	double gx1, gy1;
	double g0, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3;
	double gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2;
	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double denom;
	double dx1, dx2, dy1, dy2;
	double g_can1[ROW][COL], g_nor1[ROW][COL];
	int g_ang1[ROW][COL];
	
	double newVar = rand() % 6;

	char fileName[128];
	sprintf(fileName, "tests0_temp");
	FILE *fp;
	if((fp = fopen(fileName, "rb")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	fread(H, sizeof(double), 162 * COL * ROW, fp);
	
	
	/* Read image */
	sprintf(fileName, "tests0.pgm"); //
	load_image_file(fileName);
	defcan(g_can1);
	roberts8(g_ang1, g_nor1);
	
	if (newVar > 1.0) {
		for (y = 0 ; y < ROW ; y++) {
			for (x = 0 ; x < 27 * COL ; x++) {
					Ht[y][x] =  H[y][x + COL * 27 * 5];
			}
		}
	} else if (newVar < 1.0 / 32.0) {
		for (y = 0 ; y < ROW ; y++) {
			for (x = 0 ; x < 27 * COL ; x++) {
					Ht[y][x] =  H[y][x];
			}
		}
	} else {
		count = floor(log2(newVar)) + 5;
		for (y = 0 ; y < ROW ; y++) {
			for (x = 0 ; x < 27 * COL ; x++) {
					Ht[y][x] =  H[y][x + COL * 27 * count] + (H[y][x + COL * 27 * (count + 1)] - H[y][x + COL * 27 * count]) / (var[count + 1] - var[count]) * (newVar - var[count]);
					// printf("Ht = %f H[count] = %f H[count + 1] = %f \n", Ht[y][x], H[y][x + COL * 27 * count], H[y][x + COL * 27 * (count + 1)]);
			}
		}
	}
	
	g0 = gx2 = gy2 = 0.0;
	gx1p1 = gx1p2 = gx1p3 = gx1p4 = gy1p1 = gy1p2 = gy1p3 = gy1p4 = 0.0;
	gx1p1y1p1 = gx1p2y1p1 = gx1p3y1p1 = gx1p1y1p2 = gx1p2y1p2 = gx1p1y1p3 = 0.0;
	gx1x2 = gx1y2 = gy1x2 = gy1y2 = 0.0;
	gx1p2x2 = gx1y1y2 = gx1y1x2 = gy1p2y2 = 0.0;
	for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
		dy1 = y1 - CY;
		for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
			dx1 = x1 - CX;

			thre = (g_ang1[y1][x1] + 1) * 3 * COL;

			t0     = Ht[y1][thre + x1]           * g_can1[y1][x1];
			tx2    = Ht[y1][thre + x1 + COL]     * g_can1[y1][x1];
			ty2    = Ht[y1][thre + x1 + COL * 2] * g_can1[y1][x1];

			g0        += t0;
			gx2       += tx2;
			gy2       += ty2;
			gx1p1     += t0  * dx1;
			gx1p2     += t0  * dx1 * dx1;
			gx1p3     += t0  * dx1 * dx1 * dx1;
			gx1p4     += t0  * dx1 * dx1 * dx1 * dx1;
			gy1p1     += t0  * dy1;
			gy1p2     += t0  * dy1 * dy1;
			gy1p3     += t0  * dy1 * dy1 * dy1;
			gy1p4     += t0  * dy1 * dy1 * dy1 * dy1;
			gx1p1y1p1 += t0  * dx1 * dy1;
			gx1p2y1p1 += t0  * dx1 * dx1 * dy1;
			gx1p3y1p1 += t0  * dx1 * dx1 * dx1 * dy1;
			gx1p1y1p2 += t0  * dx1 * dy1 * dy1;
			gx1p2y1p2 += t0  * dx1 * dx1 * dy1 * dy1;
			gx1p1y1p3 += t0  * dx1 * dy1 * dy1 * dy1;
			gx1x2     += tx2 * dx1;
			gy1x2     += tx2 * dy1;
			gx1y2     += ty2 * dx1;
			gy1y2     += ty2 * dy1;
			gx1p2x2   += tx2 * dx1 * dx1;
			gx1y1y2   += ty2 * dx1 * dy1;
			gx1y1x2   += tx2 * dx1 * dy1;
			gy1p2y2   += ty2 * dy1 * dy1;
		}
	}

	#if __1000times == 1
	}
	#endif

	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;
  	printf("Time elapsed in total: %.7f ms\n",elapsed_secs);
  	
  	#if __1000times == 0
	printf("g0 = %f\n", g0);
	#endif
	
	
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
			// printf("%d %d %d\n",y,x,g_ang[y][x]);
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
	//158086 2.20739e+07
	// cout<<mean<<" "<<norm<<endl;
	mean /= (double)npo;
	norm -= (double)npo * mean * mean;
	if (norm == 0.0) norm = 1.0;
	//38.5952 1.59725e+07
	//cout<<mean<<" "<<norm<<endl;
	ratio = 1.0 / sqrt(norm);
	for (y = 0 ; y < ROW; y++) {
		for (x = 0 ; x < COL; x++) {
			g_can[y][x] = ratio * ((double)image1[y][x] - mean);
			//0.00025 -0.00966
			// printf("%.5f %.5f\n",ratio,g_can[y][x]);

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