#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "parameter.h"
#include "utility.h"
// #include"stdGpt.h"
// #include"accGpt.h"

void load_image_file(char *, unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE], int x_size1, int y_size1); /* image input */
void save_image_file(char *, unsigned char image2[MAX_IMAGESIZE][MAX_IMAGESIZE], int x_size2, int y_size2); /* image output*/

void multiplyVect3x3(double gpt[3][3], double inVect[3], double outVect[3]);
void multiplyVect4x4(double inMat[4][4], double inVect[4], double outVect[4]);
void multiplyVect8x8(double inMat[8][8], double inVect[8], double outVect[8]);
void multiply3x3(double inMat1[3][3], double inMat2[3][3], double outMat[3][3]);
void inverse3x3(double inMat[3][3], double outMat[3][3]);
void inverse4x4(double inMat[4][4], double outMat[4][4]);
void inverse8x8(double inMat[8][8], double outMat[8][8]);
void quickSortAsc(double data[], int indx[], int left, int right);
void changeValue(double *a, double *b);
void changeValueInt(int *a, int *b);

/*****************************************************************/
void multiplyVect3x3(double inMat[3][3], double inVect[3], double outVect[3]) {
	int i, j;
	double sum;
	for(i = 0 ; i < 3 ; ++i) {
		sum = 0.0;
		for(j = 0 ; j < 3 ; ++j) {
			sum += inMat[i][j] * inVect[j];
		}
		outVect[i] = sum;
	}
}

void multiplyVect4x4(double inMat[4][4], double inVect[4], double outVect[4]) {
	int i, j;
	double sum;
	for(i = 0 ; i < 4 ; ++i) {
		sum = 0.0;
		for(j = 0 ; j < 4 ; ++j) {
			sum += inMat[i][j] * inVect[j];
		}
		outVect[i] = sum;
	}
}

void multiplyVect8x8(double inMat[8][8], double inVect[8], double outVect[8]) {
	int i, j;
	double sum;
	for(i = 0 ; i < 8 ; ++i) {
		sum = 0.0;
		for(j = 0 ; j < 8 ; ++j) {
			sum += inMat[i][j] * inVect[j];
		}
		outVect[i] = sum;
	}
}

void multiply3x3(double inMat1[3][3], double inMat2[3][3], double outMat[3][3]) {
	int i, j, k;
	double sum;
	for(i = 0 ; i < 3 ; ++i) {
		for(j = 0 ; j < 3 ; ++j) {
			sum = 0.0;
			for(k = 0 ; k < 3 ; ++k) {
				sum += inMat1[i][k] * inMat2[k][j];
			}
			outMat[i][j] = sum;
		}
	}
}

void multiply8x8(double inMat1[8][8], double inMat2[8][8], double outMat[8][8]) {
	int i, j, k;
	double sum;
	for(i = 0 ; i < 8 ; ++i) {
		for(j = 0 ; j < 8 ; ++j) {
			sum = 0.0;
			for(k = 0 ; k < 8 ; ++k) {
				sum += inMat1[i][k] * inMat2[k][j];
			}
			outMat[i][j] = sum;
		}
	}
}

void inverse3x3(double inMat[3][3], double outMat[3][3]) {
	double det;
	det = inMat[0][0] * inMat[1][1] * inMat[2][2]
	                                           + inMat[1][0] * inMat[2][1] * inMat[0][2]
	                                                                                  + inMat[2][0] * inMat[0][1] * inMat[1][2]
	                                                                                                                         - inMat[0][0] * inMat[2][1] * inMat[1][2]
	                                                                                                                                                                - inMat[1][0] * inMat[0][1] * inMat[2][2]
	                                                                                                                                                                                                       - inMat[2][0] * inMat[1][1] * inMat[0][2];
	if (fabs(det) < EPS) {
		printf("3 x 3 matrix has no inverse");
		exit(1);
	}
	outMat[0][0] =  (inMat[1][1] * inMat[2][2] - inMat[2][1] * inMat[1][2]) / det;
	outMat[1][0] = -(inMat[1][0] * inMat[2][2] - inMat[2][0] * inMat[1][2]) / det;
	outMat[2][0] =  (inMat[1][0] * inMat[2][1] - inMat[2][0] * inMat[1][1]) / det;

	outMat[0][1] = -(inMat[0][1] * inMat[2][2] - inMat[2][1] * inMat[0][2]) / det;
	outMat[1][1] =  (inMat[0][0] * inMat[2][2] - inMat[2][0] * inMat[0][2]) / det;
	outMat[2][1] = -(inMat[0][0] * inMat[2][1] - inMat[2][0] * inMat[0][1]) / det;

	outMat[0][2] =  (inMat[0][1] * inMat[1][2] - inMat[1][1] * inMat[0][2]) / det;
	outMat[1][2] = -(inMat[0][0] * inMat[1][2] - inMat[1][0] * inMat[0][2]) / det;
	outMat[2][2] =  (inMat[0][0] * inMat[1][1] - inMat[1][0] * inMat[0][1]) / det;
}


void inverse4x4(double inMat[4][4], double outMat[4][4]) {
	double inMemo[4][4];
	double buf;
	int i,j,k;
	int n=4;

	for(i = 0 ; i < n ; i++){
		for(j = 0 ; j < n ; j++){
			outMat[i][j] = (i == j) ? 1.0 : 0.0;
			inMemo[i][j] = inMat[i][j];
		}
	}

	for(i = 0 ; i < n ; i++){
		buf= 1 / inMemo[i][i];
		for(j = 0 ; j < n ; j++){
			inMemo[i][j] *= buf;
			outMat[i][j] *= buf;
		}
		for(j = 0 ; j < n ; j++){
			if(i != j){
				buf = inMemo[j][i];
				for(k = 0 ; k < n ; k++){
					inMemo[j][k] -= inMemo[i][k] * buf;
					outMat[j][k] -= outMat[i][k] * buf;
				}
			}
		}
	}
}

void inverse8x8(double inMat[8][8], double outMat[8][8]) {
	double inMemo[8][8];
	double buf;
	int i,j,k;
	int n = 8;

	for(i = 0 ; i < n ; i++){
		for(j = 0 ; j < n ; j++){
			outMat[i][j] = (i == j) ? 1.0 : 0.0;
			inMemo[i][j] = inMat[i][j];
		}
	}

	for(i = 0 ; i < n ; i++){
		buf= 1 / inMemo[i][i];
		for(j = 0 ; j < n ; j++){
			inMemo[i][j] *= buf;
			outMat[i][j] *= buf;
		}
		for(j = 0 ; j < n ; j++){
			if(i != j){
				buf = inMemo[j][i];
				for(k = 0 ; k < n ; k++){
					inMemo[j][k] -= inMemo[i][k] * buf;
					outMat[j][k] -= outMat[i][k] * buf;
				}
			}
		}
	}
}

void load_image_file(char *filename, unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE], int x_size1, int y_size1) {
    /* Input of header & body information of pgm file */
    /* for image1[ ][ ] */
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

void save_image_file(char *filename, unsigned char image2[MAX_IMAGESIZE][MAX_IMAGESIZE], int x_size2, int y_size2) {
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

void changeValue(double *a, double *b) {
    double temp;
    temp = *a;
    *a   = *b;
    *b   = temp;
}

void changeValueInt(int *a, int *b) {
    int temp;
    temp = *a;
    *a   = *b;
    *b   = temp;
}

void quickSortAsc(double data[], int indx[], int left, int right) {
    int i = left, j = right, idx;
    double maxData = data[right], minData = data[left];
    for (idx = left ; idx <= right ; idx++) {
        if (data[idx] < minData) minData = data[idx];
        if (data[idx] > maxData) maxData = data[idx];
    }
    
    double pivot; // = 0.5 * (data[left] + data[right]);
    if (minData != maxData) {
        pivot = 0.5 * (minData + maxData);
    } else {
        return;
    }
    // double thres = data[left];
    // printf("thres = %f \n", pivot);
    if (i < j) {
        while (i <= j) {
            while (i <= right && data[i] <  pivot) i++;
            while (j >= left  && data[j] >= pivot) j--;
            // printf("i = %d, j = %d\n", i, j);
            if (i > j) break;
            changeValue(&data[i], &data[j]);
            changeValueInt(&indx[i], &indx[j]);
            i++;
            j--;
        }
        quickSortAsc(data, indx, left,  i - 1);
        quickSortAsc(data, indx, j + 1, right);
    }
}
