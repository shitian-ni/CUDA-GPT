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
#include "stdGpt.h"
// #include "gpuGpt.h"

void  procImg(double g_can[ROW][COL], int g_ang[ROW][COL], double g_nor[ROW][COL], char g_HoG[ROW][COL][8], char sHoG[ROW - 4][COL - 4], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]);
void  roberts8(int g_ang[ROW][COL], double g_nor[ROW][COL], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]);
void  smplHoG64(char sHoG[ROW - 4][COL - 4], int g_ang[ROW][COL], double g_nor[ROW][COL]);
void  defcan(double g_can[ROW][COL], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]);
void  defcan2(double g_can[ROW - CANMARGIN][COL - CANMARGIN], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]);
void  calHoG(int g_ang[ROW][COL], char g_HoG[ROW][COL][8]);

double winpat(int g_ang1[ROW][COL], int g_ang2[ROW][COL]);
double sHoGpat(char sHoG1[ROW - 4][COL - 4], char sHoG2[ROW - 4][COL - 4]);
int    sHoG2Idx(char sHoG);

void initGpt(double gpt[3][3]);
void initGpt2(double gpt[3][3], double alpha, double beta, double b1, double b2, double rotation);	// Use non elemental matrix as initial condition
void copyGpt(double inGpt[3][3], double outGpt[3][3]);
void bilinear_normal_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
		unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE], unsigned char image2[MAX_IMAGESIZE][MAX_IMAGESIZE]);
void bilinear_normal_inverse_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
		unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE], unsigned char image2[MAX_IMAGESIZE][MAX_IMAGESIZE]);

void gatcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
            int g_ang2[ROW][COL], double g_can2[ROW][COL],
            double gwt[ROW][COL], double gpt[3][3]);
void ngatcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
             int g_ang2[ROW][COL], double g_can2[ROW][COL],
             double gwt[ROW][COL], double gpt[3][3], double dnn);
void pptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
            int g_ang2[ROW][COL], double g_can2[ROW][COL],
            double gwt[ROW][COL], double gpt[3][3]);
void npptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
             int g_ang2[ROW][COL], double g_can2[ROW][COL],
             double gwt[ROW][COL], double gpt[3][3], double dnn);
void sgptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
            int g_ang2[ROW][COL], double g_can2[ROW][COL],
            double gwt[ROW][COL], double gpt[3][3]);
void nsgptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
             int g_ang2[ROW][COL], double g_can2[ROW][COL],
             double gwt[ROW][COL], double gpt[3][3], double dnn);
void nsgptcorSpHOG5x5(int g_ang1[ROW][COL], char sHoG1[ROW - 4][COL - 4], double g_can1[ROW][COL],
		int g_ang2[ROW][COL], char sHoG2[ROW - 4][COL - 4], double g_can2[ROW][COL],
		double gwt[ROW][COL], double gpt[3][3], double dnn);

/**************************************************************************/
void  procImg(double g_can[ROW][COL], int g_ang[ROW][COL], double g_nor[ROW][COL], char g_HoG[ROW][COL][8], char sHoG[ROW - 4][COL - 4], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE])
{
#if isGPU == 0
	defcan2(g_can, image1);				/* canonicalization */
	roberts8(g_ang, g_nor, image1);		/* 8-quantization of gradient dir */
	calHoG(g_ang, g_HoG);				/* calculate sHOG pattern */
	smplHoG64(sHoG, g_ang, g_nor);		/* Numberring the sHOG pattern to sHoGNUMBER */
#elif isGPU == 1
    // Shitian NI
    defcan2(g_can, image1);				/* canonicalization */
    cout<<"CPU: "<<endl;
    for(int i=5;i<7;i++){
    	for(int j=5;j<7;j++){
    		cout<<g_can[i][j]<<" ";
    	}
    	cout<<endl;
    }
    cuda_calc_defcan1(image1,g_can);
    cout<<"GPU"<<endl;
    for(int i=5;i<7;i++){
    	for(int j=5;j<7;j++){
    		cout<<g_can[i][j]<<" ";
    	}
    	cout<<endl;
    }
	roberts8(g_ang, g_nor, image1);		/* 8-quantization of gradient dir */
	calHoG(g_ang, g_HoG);				/* calculate sHOG pattern */
	smplHoG64(sHoG, g_ang, g_nor);		/* Numberring the sHOG pattern to sHoGNUMBER */
#endif
}

void roberts8(int g_ang[ROW][COL], double g_nor[ROW][COL], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]) {
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

void smplHoG64(char sHoG[ROW - 4][COL - 4], int g_ang[ROW][COL], double g_nor[ROW][COL]){
    int x, y, dx, dy, dir;
    double HoG[8];
    int HoGIdx[8];

    for (y = 0 ; y < ROW - 4 ; y++) {
        for (x = 0 ; x < COL - 4 ; x++) {
            sHoG[y][x] = -1;
            // initialize
            for (dir = 0 ; dir < 8 ; dir++) {
                HoG[dir] = 0.0;
                HoGIdx[dir] = dir + 1;
            }
            // calculate HoG
            for (dy = y ; dy < y + 5 ; dy++) {
                for (dx = x ; dx < x + 5 ; dx++) {
                    if (g_ang[dy][dx] == -1) break;
                    HoG[g_ang[dy][dx]] += g_nor[dy][dx];
                }
            }
            // sort of the 8 HoG (One step of bubble sort)
            for (dir = 7 ; dir > 0 ; dir--) {
                if (HoG[dir] >= HoG[dir - 1]) {
                    changeValue(&HoG[dir], &HoG[dir - 1]);
                    changeValueInt(&HoGIdx[dir], &HoGIdx[dir - 1]);
                }
            }
            for (dir = 7 ; dir > 1 ; dir--) {
                if (HoG[dir] >= HoG[dir - 1]) {
                    changeValue(&HoG[dir], &HoG[dir - 1]);
                    changeValueInt(&HoGIdx[dir], &HoGIdx[dir - 1]);
                }
            }
            // calculate the direction
            if (HoG[0] > SHoGTHRE) {
                sHoG[y][x] = (char) HoGIdx[0];
                if (HoG[1] > SHoGSECONDTHRE * HoG[0]) {
                    sHoG[y][x] = sHoG[y][x] * 10 + (char) HoGIdx[1];
                }
            }
            // printf("<%d, %d> HoG = %d \n", y, x, sHoG[y][x]);
        }
    }
}

void defcan(double g_can[ROW][COL], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]) {
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

void defcan2(double g_can[ROW - CANMARGIN][COL - CANMARGIN], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]) {
	/* definite canonicalization */
	int x, y;
	double mean, norm, ratio; // mean: mean value, norm: normal factor, ratio:
	int margine = CANMARGIN / 2;
	int npo; // number of point

	// npo = (ROW - 2 * MARGINE) * (COL - 2 * MARGINE);
	npo = 0;
	mean = norm = 0.0;
	for (y = margine ; y < ROW - margine ; y++) {
		for (x = margine ; x < COL - margine ; x++) {
			if (image1[y][x] != WHITE) {
				mean += (double)image1[y][x];
				norm += (double)image1[y][x] * (double)image1[y][x];
				npo++;
			}
		}
	}
	mean /= (double)npo;
	norm -= (double)npo * mean * mean;
	if (norm == 0.0) norm = 1.0;
	ratio = 1.0 / sqrt(norm);
	for (y = 0 ; y < ROW - CANMARGIN; y++) {
		for (x = 0 ; x < COL - CANMARGIN; x++) {
			if (image1[y][x] != WHITE) {
				g_can[y][x] = ratio * ((double)image1[y][x] - mean);
			} else {
				g_can[y][x] = 0.0;
			}
		}
	}
}

void calHoG(int g_ang[ROW][COL], char g_HoG[ROW][COL][8]) {
	int x, y, dir;
	int tx, ty;
	// initialize
	for (y = 0 ; y < ROW ; y++) {
		for (x = 0 ; x < COL ; x++) {
			for (dir = 0 ; dir < 8 ; dir++) {
				g_HoG[y][x][dir] = 0;
			}
		}
	}

	for (y = 2 ; y < ROW - 2 ; y++) {
		for (x = 2 ; x < COL - 2 ; x++) {
			// scan 5x5 block
			for (ty = y - 2 ; ty < y + 3 ; ty++) {
				for (tx = x - 2 ; tx < x + 3 ; tx++) {
					if (g_ang[ty][tx] == -1) {
						continue;
					} else {
						g_HoG[y][x][g_ang[ty][tx]]++;
					}
				}
			}
			// scan 5x5 block finish
		}
	}

}

double winpat(int g_ang1[ROW][COL], int g_ang2[ROW][COL]) {
	/* calculation of mean of nearest-neighbor interpoint distances */
	/* with the same angle code between two images */
	double min, minInit, delta, dnn1, dnn2;
	int x1, y1, x2, y2;
	int angcode;
	int count1, count2;

	minInit = (ROW - 2 * MARGINE) * (ROW - 2 * MARGINE) + (COL - 2 * MARGINE) * (COL - 2 * MARGINE);
	/* from the 1st image */
	count1 = 0;
	dnn1 = 0.0;
	for (y1 = MARGINE ; y1 < ROW - MARGINE; y1++) {
		for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
			angcode = g_ang1[y1][x1];
			if (angcode == -1) continue;
			count1++;
			min = minInit;
			for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
				for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
					if (g_ang2[y2][x2] != angcode) continue;
					delta = (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1);
					if (delta < min) min = delta;
				}
			}
			// printf("angCode = %d, (%d, %d) nn1 = %f \n", angcode, x1, y1, sqrt(min));
			dnn1 += sqrt(min);
		}
	}
	dnn1 /= (double)count1;
	// printf("  count1  %d  ", count1);

	/* from the 2nd image */
	count2 = 0;
	dnn2 = 0.0;
	for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
		for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
			angcode = g_ang2[y2][x2];
			if (angcode == -1) continue;
			count2++;
			min = minInit;
			for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
				for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
					if (g_ang1[y1][x1] != angcode) continue;
					delta = (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1);
					if (delta < min) min = delta;
				}
			}
			dnn2 += sqrt(min);
		}
	}
	dnn2 /= (double)count2;
	// printf("  count2  %d  ", count2);

	/* printf("Gauss parameter %f  %f  \n", dnn1, dnn2); */
	return (dnn1 + dnn2)/2.0;
}

double sHoGpat(char sHoG1[ROW - 4][COL - 4], char sHoG2[ROW - 4][COL - 4]) {
	/* calculation of mean of nearest-neighbor interpoint distances */
	/* with the same angle code between two images */
	double min, minInit, delta, dnn1, dnn2;
	int x1, y1, x2, y2;
	int angcode;
	int count1, count2;
	int margin = 2;

	minInit = (ROW - 2 * margin) * (ROW - 2 * margin) + (COL - 2 * margin) * (COL - 2 * margin);
	/* from the 1st image */
	count1 = 0;
	dnn1 = 0.0;
	for (y1 = margin ; y1 < ROW - margin; y1++) {
		for (x1 = margin ; x1 < COL - margin ; x1++) {
			angcode = sHoG1[y1 - margin][x1 - margin];
			if (angcode == -1) continue;
			count1++;
			min = minInit;
			for (y2 = margin ; y2 < ROW - margin ; y2++) {
				for (x2 = margin ; x2 < COL - margin ; x2++) {
					// if (angcode < 10) {
						if (sHoG2[y2 - margin][x2 - margin] != angcode) continue;
					// } else {
						// if (sHoG2[y2 - margin][x2 - margin] != angcode && sHoG2[y2 - margin][x2 - margin] != 10 * (angcode % 10) + (angcode / 10)) continue;
					// }
					delta = (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1);
					if (delta < min) min = delta;
				}
			}
			// printf("angCode = %d, (%d, %d) nn1 = %f \n", angcode, x1, y1, sqrt(min));
			dnn1 += sqrt(min);
			// printf("angCode = %d, (%d, %d) nn1 = %f \n", angcode, x1, y1, dnn1);
		}
	}
	dnn1 /= (double)count1;
	// printf("  count1  %d , dnn1  %f \n", count1, dnn1);

	/* from the 2nd image */
	count2 = 0;
	dnn2 = 0.0;
	for (y2 = margin ; y2 < ROW - margin ; y2++) {
		for (x2 = margin ; x2 < COL - margin ; x2++) {
			angcode = sHoG2[y2 - margin][x2 - margin];
			if (angcode == -1) continue;
			count2++;
			min = minInit;
			for (y1 = margin ; y1 < ROW - margin ; y1++) {
				for (x1 = margin ; x1 < COL - margin ; x1++) {
					if (angcode < 10) {
						if (sHoG1[y1 - margin][x1 - margin] != angcode) continue;
					} else {
						// if (sHoG1[y1 - margin][x1 - margin] != angcode && sHoG1[y1 - margin][x1 - margin] != 10 * (angcode % 10) + (angcode / 10)) continue;
						if (sHoG1[y1 - margin][x1 - margin] != angcode) continue;
					}
					delta = (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1);
					if (delta < min) min = delta;
				}
			}
			// printf("angCode = %d, (%d, %d) nn1 = %f \n", angcode, x1, y1, sqrt(min));
			dnn2 += sqrt(min);
		}
	}
	dnn2 /= (double)count2;
	// printf("  count2  %d , dnn2  %f  \n", count2, dnn2);

	// printf("Gauss parameter %f  %f  \n", dnn1, dnn2);
	return (dnn1 + dnn2)/2.0;
}

int sHoG2Idx(char sHoG) {
	int tempValue = sHoG;
	int quot, remd;
	if (tempValue < 10) return tempValue - 1;

	quot = tempValue / 10; remd = tempValue % 10;
	if (quot > remd)
		return (8 + (quot - 1) * 7 + remd) - 1;
	else
		return (8 + (quot - 1) * 7 + remd - 1) - 1;

}

void initGpt(double gpt[3][3]) {
	gpt[0][0] = 1.0; gpt[0][1] = 0.0; gpt[1][0] = 0.0; gpt[1][1] = 1.0;
	gpt[0][2] = 0.0; gpt[1][2] = 0.0;
	gpt[2][0] = 0.0; gpt[2][1] = 0.0;
	gpt[2][2] = 1.0;
}

void initGpt2(double gpt[3][3], double alpha, double beta, double b1, double b2, double rotation) {
	int x, y;
	double tmp[3][3], gpt_[3][3];

	for (x = 0 ; x < 3 ; x++)
		for (y = 0 ; y < 3 ; y++)
			gpt_[x][y] = 0.0;

	gpt[0][0] = alpha, gpt[0][1] = 0.0; gpt[1][0] = 0.0; gpt[1][1] = beta;
	gpt[0][2] = b1; gpt[1][2] = b2;
	gpt[2][0] = 0.0; gpt[2][1] = 0.0;
	gpt[2][2] = 1.0;

	tmp[0][0] =  cos(rotation * PI / 180.0), tmp[0][1] = sin(rotation * PI / 180.0);
	tmp[1][0] = -sin(rotation * PI / 180.0), tmp[1][1] = cos(rotation * PI / 180.0);
	tmp[0][2] = tmp[1][2] = tmp[2][0] = tmp[2][1] = 0.0;
	tmp[2][2] = 1.0;

	multiply3x3(tmp, gpt, gpt_);
	inverse3x3(gpt_, gpt);
	// print3x3(gpt_);
	// print3x3(gpt);
}

void copyGpt(double inGpt[3][3], double outGpt[3][3]) {
	int i, j;
	for(i = 0 ; i < 3 ; ++i) {
		for(j = 0 ; j < 3 ; ++j) {
			outGpt[i][j] = inGpt[i][j];
		}
	}
}

void bilinear_normal_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
		unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE], unsigned char image2[MAX_IMAGESIZE][MAX_IMAGESIZE]) {
	/* projection transformation of the image by bilinear interpolation */
	double inv_gpt[3][3];
	inverse3x3(gpt, inv_gpt);
	bilinear_normal_inverse_projection(inv_gpt, x_size1, y_size1, x_size2, y_size2, image1, image2);
}

void bilinear_normal_inverse_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
		unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE], unsigned char image2[MAX_IMAGESIZE][MAX_IMAGESIZE]) {
	/* inverse projection transformation of the image by bilinear interpolation */
	int    x, y;
	double inVect[3], outVect[3];
	double x_new, y_new, x_frac, y_frac;
	double gray_new;
	int m, n;
	int cx, cy, cx2, cy2;

	/* output image generation by bilinear interpolation */
	// x_size2 = x_size1;
	// y_size2 = y_size1;
	if (y_size1 == ROW) {
		cx  = CX,  cy  = CY;
		cx2 = CX2, cy2 = CY2;
	} else {
		cx  = CX2, cy  = CY2;
		cx2 = CX,  cy2 = CY;
	}
	inVect[2] = 1.0;
	for (y = 0 ; y < y_size1; y++) {
		inVect[1] = y - cy;
		for (x = 0 ; x < x_size1; x++) {
			inVect[0] = x - cx;
			multiplyVect3x3(gpt, inVect, outVect);
			x_new = outVect[0] / outVect[2] + cx2;
			y_new = outVect[1] / outVect[2] + cy2;
			m = (int)floor(x_new);
			n = (int)floor(y_new);
			x_frac = x_new - m;
			y_frac = y_new - n;

			if (m >= 0 && m+1 < x_size2 && n >= 0 && n+1 < y_size2) {
				gray_new = (1.0 - y_frac) * ((1.0 - x_frac) * image1[n][m] + x_frac * image1[n][m+1])
	  		 + y_frac * ((1.0 - x_frac) * image1[n+1][m] + x_frac * image1[n+1][m+1]);
				image2[y][x] = (unsigned char)gray_new;
			} else {
#ifdef BACKGBLACK
				image2[y][x] = BLACK;
#else
				image2[y][x] = WHITE;
#endif
			}
			//printf("(%d %d): inVect = (%f; %f; %f), outVect = (%f; %f; %f) \n", y, x, inVect[0], inVect[1], inVect[2], outVect[0], outVect[1], outVect[2]);
			//printf("%d %d = %d %d || n = %d, m = %d, x_new = %f, y_new = %f, x_frac = %f, y_frac = %f \n", y, x, image2[y][x], image1[y][x], n, m, x_new, y_new, x_frac, y_frac);
		}
	}
}

/**************************GPT*************************/
void gatcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
		int g_ang2[ROW][COL], double g_can2[ROW][COL],
		double gwt[ROW][COL], double gpt[3][3]) {
	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2;
	
	double* g = (double*) malloc (G_NUM * sizeof(double));
	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double det, U11, U12, U21, U22;
	double W11, W12, W21, W22, Uinv11, Uinv12, Uinv21, Uinv22;
	double dx1, dx2, dy1, dy2;
	double V11, V12, V21, V22, VTinv11, VTinv12, VTinv21, VTinv22;
	double tGpt1[3][3], tGpt2[3][3];

	int count = 0;
	/* Gaussian weigthed mean values */
	g0 = gx1 = gy1 = gx2 = gy2 = 0.0;
	gx1x1 = gx1y1 = gy1y1 = gx1x2 = gx1y2 = gy1x2 = gy1y2 = 0.0;
	gx2x2 = gx2y2 = gy2y2 = 0.0;
	for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
		//printf("y1 = %d; \n", y1);
		dy1 = y1 - CY;
		for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
			dx1 = x1 - CX;
			t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
			for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
				dy2 = y2 - CY;
				for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
					dx2 = x2 - CX;
					if (g_ang1[y1][x1] == g_ang2[y2][x2] && g_ang1[y1][x1] != -1) {
						/*
#ifdef LABINTRO // save position data
						fprintf(fpp2, "%d,%f,%f,%d,%d,%d,%d\n", 0, 0.0, 0.0, x1, y1, x2, y2);
#endif
						*/
						tv   = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can1[y1][x1] * g_can2[y2][x2];
						t0    += tv;
						tx2   += tv * dx2;
						ty2   += tv * dy2;
						gx2x2 += tv * dx2 * dx2;
						gx2y2 += tv * dx2 * dy2;
						gy2y2 += tv * dy2 * dy2;
						/*
						if (count < 100 & x1 == 1){
							printf("x1  = %2d, y1  = %2d, x2  = %2d, y2  = %2d \n", y1 , x1 , y2 , x2);
							printf("dx1 = %2.0f, dy1 = %2.0f, dx2 = %2.0f, dy2 = %2.0f \n", dy1, dx1, dy2, dx2);
							printf("tv = %10.8f \n", tv);
							printf("t0 = %10.8f \n", t0);
							count++;
						}
						*/
					}
				}
			}
			//printf("t0 = %f tx2 = %f ty2 = %f \n", t0 / g_can1[y1][x1], tx2 / g_can1[y1][x1], ty2 / g_can1[y1][x1]);
			g0    += t0;
			gx2   += tx2;
			gy2   += ty2;
			gx1   += t0  * dx1;
			gy1   += t0  * dy1;
			gx1x1 += t0  * dx1 * dx1;
			gx1y1 += t0  * dx1 * dy1;
			gy1y1 += t0  * dy1 * dy1;
			gx1x2 += tx2 * dx1;
			gx1y2 += ty2 * dx1;
			gy1x2 += tx2 * dy1;
			gy1y2 += ty2 * dy1;
		}
	}

	/*
	printf("gx1x1 = %f gx1x2 = %f | gy1y1 = %f gy1y2 = %f\n", gx1x1/g0, gx1x2/g0, gy1y1/g0, gy1y2/g0);
	printf("gx1y1 = %f gx1y2 = %f | gy1x2 = %f gy1y2 = %f\n", gx1y1/g0, gx1y2/g0, gy1x2/g0, gy1y2/g0);
	printf("g0 = %f\n", g0);


	printf("gx1x1 = %f gx1x2 = %f | gy1y1 = %f gy1y2 = %f\n", gx1x1, gx1x2, gy1y1, gy1y2);
	printf("gx1y1 = %f gx1y2 = %f | gy1x2 = %f gy1y2 = %f\n", gx1y1, gx1y2, gy1x2, gy1y2);
	printf("g0 = %f gx2 = %f gy2 = %f | gx1 = %f gy1 = %f\n", g0, gx2, gy2, gx1, gy1);
	*/

	if (g0 == 0.0) {
		printf("GAT calculation failure by zero sum!!!\n");
		return;
	}
	/* V = bar(x_2 x_1^T) - bar(x_2) bar(x_1)^T / var(1) */
	V11 = gx1x2 - gx1 * gx2 / g0;
	V12 = gy1x2 - gy1 * gx2 / g0;
	V21 = gx1y2 - gx1 * gy2 / g0;
	V22 = gy1y2 - gy1 * gy2 / g0;

	/* U = bar(x_1 x_1^T) - bar(x_1) bar(x_1)^T / bar(1) */
	U11       = gx1x1 - gx1 * gx1 / g0;
	U12 = U21 = gx1y1 - gy1 * gx1 / g0;
	U22       = gy1y1 - gy1 * gy1 / g0;

	/* Uinv = U^{-1} */
	det = U11 * U22 - U12 * U21;
	if (fabs(det) < EPS) {
		printf("GAT calculation failure by zero det of U !!! det = %f\n", det);
		return;
	}
	Uinv11          =  U22 / det;
	Uinv12 = Uinv21 = -U12 / det;
	Uinv22          =  U11 / det;

	/* bar(x_2 x_2^T) - bar(x_2) bar(x_2)^T / bar(1) */
	W11       = gx2x2 - gx2 * gx2 / g0;
	W12 = W21 = gx2y2 - gx2 * gy2 / g0;
	W22       = gy2y2 - gy2 * gy2 / g0;

	tGpt1[0][0] = V11 * Uinv11 + V12 * Uinv21;
	tGpt1[0][1] = V11 * Uinv12 + V12 * Uinv22;
	tGpt1[1][0] = V21 * Uinv11 + V22 * Uinv21;
	tGpt1[1][1] = V21 * Uinv12 + V22 * Uinv22;

	tGpt1[0][2] = (gx2 - tGpt1[0][0] * gx1 - tGpt1[0][1] * gy1) / g0;
	tGpt1[1][2] = (gy2 - tGpt1[1][0] * gx1 - tGpt1[1][1] * gy1) / g0;

	tGpt1[2][0] = tGpt1[2][1] = 0.0;
	tGpt1[2][2] = 1.0;

	/* printf("datr a11 = %f, a12 = %f, a21 = %f, a22 = %f, b1 = %f b2 = %f\n", a1[0][0], a1[0][1], a1[1][0], a1[1][1], b1[0], b1[1]); */

	/* update of GAT components */
	multiply3x3(tGpt1, gpt, tGpt2);
	copyGpt(tGpt2, gpt);
}

void pptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
		int g_ang2[ROW][COL], double g_can2[ROW][COL],
		double gwt[ROW][COL], double gpt[3][3]) {
	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2;
	double* g = (double*) malloc (G_NUM * sizeof(double));
	double tv, t0, tx2, ty2;
	double det, U11, U12, U21, U22, Uinv11, Uinv12, Uinv21, Uinv22, v1, v2;
	double dx1, dx2, dy1, dy2;
	double tIn12, tIn11, gIn12x1x1, gIn12x1y1, gIn12y1y1, gIn11x1, gIn11y1, gIn12x1, gIn12y1;
	double tGpt1[3][3], tGpt2[3][3];
	double gIn11, gIn12, gIn22;

	/* Gaussian weigthed mean values */
	gIn12x1x1 = gIn12x1y1 = gIn12y1y1 = gIn11x1 = gIn11y1 = gIn12x1 = gIn12y1 = 0.0;
	gIn11 = gIn12 = gIn22 = 0.0;
	for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
		dy1 = y1 - CY;
		for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
			dx1 = x1 - CX;
			t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
			for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
				dy2 = y2 - CY;
				for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
					dx2 = x2 - CX;
					if (g_ang1[y1][x1] == g_ang2[y2][x2] && g_ang1[y1][x1] != -1) {
						tv   = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can1[y1][x1] * g_can2[y2][x2];
						t0    += tv;
						tx2   += tv * dx2;
						ty2   += tv * dy2;
						gIn22 += tv * (dx2 * dx2 + dy2 * dy2);
					}
				}
			}
			g0        += t0;
			tIn12      =  dx1 * tx2 + dy1 * ty2;
			tIn11      = (dx1 * dx1 + dy1 * dy1) * t0;
			gIn12x1x1 += tIn12 * dx1 * dx1;
			gIn12x1y1 += tIn12 * dx1 * dy1;
			gIn12y1y1 += tIn12 * dy1 * dy1;
			gIn11x1   += tIn11 * dx1;
			gIn11y1   += tIn11 * dy1;
			gIn12x1   += tIn12 * dx1;
			gIn12y1   += tIn12 * dy1;
			gIn11     += (dx1 * dx1 + dy1 * dy1) * t0;
			gIn12     += dx1 * tx2 + dy1 * ty2;
		}
	}

	// printf("gx1x1 = %f gx1x2 = %f | gy1y1 = %f gy1y2 = %f\n", gx1x1/g0, gx1x2/g0, gy1y1/g0, gy1y2/g0);

	if (g0 == 0.0) {
		printf("PPT calculation failure by zero sum!!!\n");
		return;
	}

	U11 =       gIn12x1x1;
	U12 = U21 = gIn12x1y1;
	U22 =       gIn12y1y1;

	/* U^{-1} */
	det = U11 * U22 - U21 * U12;
	if (fabs(det) < EPS) {
		printf("PPT calculation failure by zero det of V for Both side !!! det = %f\n", det);
		return;
	}
	Uinv11 =  U22 / det;
	Uinv12 = -U12 / det;
	Uinv21 = -U21 / det;
	Uinv22 =  U11 / det;

	/* V = bar(<x_1 x_1> x_1) - bar(<x_1 x_2> x_1) */
	v1 = gIn11x1 - gIn12x1;
	v2 = gIn11y1 - gIn12y1;

	tGpt1[0][0] = 1.0; tGpt1[0][1] = 0.0; tGpt1[0][2] = 0.0;
	tGpt1[1][0] = 0.0; tGpt1[1][1] = 1.0; tGpt1[1][2] = 0.0;

	tGpt1[2][0] = Uinv11 * v1 + Uinv12 * v2;
	tGpt1[2][1] = Uinv21 * v1 + Uinv22 * v2;

	tGpt1[2][2] = 1.0;

	// printf("c1 = %f  c2 = %f \n", tGpt1[2][0], tGpt1[2][1]);
	// printf("gIn11 = %f  gIn12 = %f  gIn22 = %f \n", gIn11, gIn12, gIn22);

	/* update of GAT components */
	multiply3x3(tGpt1, gpt, tGpt2);
	copyGpt(tGpt2, gpt);
}

void ngatcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
		int g_ang2[ROW][COL], double g_can2[ROW][COL],
		double gwt[ROW][COL], double gpt[3][3], double dnn) {
	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2;
	// double g0, gx1, gy1, gx2, gy2, gx1x1, gx1y1, gy1y1, gx1x2, gx1y2, gy1x2, gy1y2;
	double* g = (double*) malloc (G_NUM * sizeof(double));

	double tv, t0, tx2, ty2;
	double det, U11, U12, U21, U22;
	double Uinv11, Uinv12, Uinv21, Uinv22;
	double dx1, dx2, dy1, dy2;
	double V11, V12, V21, V22, VTinv11, VTinv12, VTinv21, VTinv22;
	double tGpt1[3][3], tGpt2[3][3];

	int count = 0;
	/* Gaussian weigthed mean values */
	g0 = gx1 = gy1 = gx2 = gy2 = 0.0;
	gx1x1 = gx1y1 = gy1y1 = gx1x2 = gx1y2 = gy1x2 = gy1y2 = 0.0;
	for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
		dy1 = y1 - CY;
		for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
			dx1 = x1 - CX;
			t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
			for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
				dy2 = y2 - CY;
				for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
					dx2 = x2 - CX;
					if (g_ang1[y1][x1] == g_ang2[y2][x2] && g_ang1[y1][x1] != -1) {
						tv   = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can1[y1][x1] * g_can2[y2][x2];
						t0    += tv;
						tx2   += tv * dx2;
						ty2   += tv * dy2;
					}
				}
			}
			g0    += t0;
			gx2   += tx2;
			gy2   += ty2;
			gx1   += t0  * dx1;
			gy1   += t0  * dy1;
			gx1x1 += t0  * dx1 * dx1;
			gx1y1 += t0  * dx1 * dy1;
			gy1y1 += t0  * dy1 * dy1;
			gx1x2 += tx2 * dx1;
			gx1y2 += ty2 * dx1;
			gy1x2 += tx2 * dy1;
			gy1y2 += ty2 * dy1;
		}
	}

	// printf("gx1x1 = %f gx1x2 = %f | gy1y1 = %f gy1y2 = %f\n", gx1x1/g0, gx1x2/g0, gy1y1/g0, gy1y2/g0);
	// printf("gx1y1 = %f gx1y2 = %f | gy1x2 = %f gy1y2 = %f\n", gx1y1/g0, gx1y2/g0, gy1x2/g0, gy1y2/g0);
	// printf("g0 = %f\n", g0);

	if (g0 == 0.0) {
		printf("GAT calculation failure by zero sum!!!\n");
		return;
	}
	/* V = bar(x_2 x_1^T) - bar(x_2) bar(x_1)^T / var(1) */
	V11 = gx1x2 - gx1 * gx2 / g0;
	V12 = gy1x2 - gy1 * gx2 / g0;
	V21 = gx1y2 - gx1 * gy2 / g0;
	V22 = gy1y2 - gy1 * gy2 / g0;

	/* U = bar(x_1 x_1^T) - bar(x_1) bar(x_1)^T / bar(1) */
	U11       = gx1x1 - gx1 * gx1 / g0;
	U12 = U21 = gx1y1 - gy1 * gx1 / g0;
	U22       = gy1y1 - gy1 * gy1 / g0;

	/* Uinv = U^{-1} */
	det = U11 * U22 - U12 * U21;
	if (fabs(det) < EPS) {
		printf("GAT calculation failure by zero det of U !!! det = %f\n", det);
		return;
	}
	Uinv11          =  U22 / det;
	Uinv12 = Uinv21 = -U12 / det;
	Uinv22          =  U11 / det;

	double r = pow(WGT * dnn, 2) / 2 * g0;
	double T11, T12, T21, T22;
	double M[4][4];
	double Minv[4][4];
	double m[4], dGpt[4];

	/*
	for (x1 = 0; x1 < 3; x1++) {
		for (y1 = 0; y1 < 3; y1++) {
			tGpt1[x1][y1] = (x1 == y1) ? 1.0 : 0.0;
		}
	}
	*/

	tGpt1[0][0] = V11 * Uinv11 + V12 * Uinv21;
	tGpt1[0][1] = V11 * Uinv12 + V12 * Uinv22;
	tGpt1[1][0] = V21 * Uinv11 + V22 * Uinv21;
	tGpt1[1][1] = V21 * Uinv12 + V22 * Uinv22;

	// Newton-Raphson
	int iter;	// loop variable
	double zeroConfirm[2][2];
	double residual = 100.0;
	double res = 0.0;
	for (iter = 0; iter < MAXNR; iter++) {
		det = tGpt1[0][0] * tGpt1[1][1] - tGpt1[1][0] * tGpt1[0][1];
		T11 =  tGpt1[1][1] / det;
		T12 = -tGpt1[0][1] / det;
		T21 = -tGpt1[1][0] / det;
		T22 =  tGpt1[0][0] / det;

		zeroConfirm[0][0] = tGpt1[0][0] * U11 + tGpt1[0][1] * U21 - V11 - r * T11;
		zeroConfirm[0][1] = tGpt1[0][0] * U12 + tGpt1[0][1] * U22 - V12 - r * T12;
		zeroConfirm[1][0] = tGpt1[1][0] * U11 + tGpt1[1][1] * U21 - V21 - r * T21;
		zeroConfirm[1][1] = tGpt1[1][0] * U12 + tGpt1[1][1] * U22 - V22 - r * T22;

		res = sqrt(zeroConfirm[0][0] * zeroConfirm[0][0] + zeroConfirm[0][1] * zeroConfirm[0][1] + zeroConfirm[1][0] * zeroConfirm[1][0] + zeroConfirm[1][1] * zeroConfirm[1][1]);
		//printf("zero11 = %6.3f, zero12 = %6.3f, zero21 = %6.3f, zero22 = %6.3f \n", zeroConfirm[0][0], zeroConfirm[0][1], zeroConfirm[1][0], zeroConfirm[1][1]);
		//printf("res = %1.5f\n", res);

		if (EPS2 > res)// || residual < res)
			break;
		else
			residual = res;

		M[0][0] = U11 + r * T11 * T11;
		M[0][1] = U21 + r * T11 * T21;
		M[0][2] =           T12 * T11;
		M[0][3] =       r * T12 * T21;

		M[1][0] = U12 + r * T11 * T12;
		M[1][1] = U22 + r * T11 * T22;
		M[1][2] =           T12 * T12;
		M[1][3] =       r * T12 * T22;

		M[2][0] =       r * T21 * T11;
		M[2][1] =       r * T21 * T21;
		M[2][2] = U11 +     T22 * T11;
		M[2][3] = U21 + r * T22 * T21;

		M[3][0] =       r * T21 * T12;
		M[3][1] =       r * T21 * T22;
		M[3][2] = U12 +     T22 * T12;
		M[3][3] = U22 + r * T22 * T22;

		m[0] = -U11 * tGpt1[0][0] - U21 * tGpt1[0][1] + V11 + r * T11;
		m[1] = -U12 * tGpt1[0][0] - U22 * tGpt1[0][1] + V12 + r * T12;
		m[2] = -U11 * tGpt1[1][0] - U21 * tGpt1[1][1] + V21 + r * T21;
		m[3] = -U12 * tGpt1[1][0] - U22 * tGpt1[1][1] + V22 + r * T22;

		inverse4x4(M, Minv);
		multiplyVect4x4(Minv, m, dGpt);

		tGpt1[0][0] += dGpt[0];
		tGpt1[0][1] += dGpt[1];
		tGpt1[1][0] += dGpt[2];
		tGpt1[1][1] += dGpt[3];

		// printf("dAxx = %10.6f, dAxy = %10.6f, dAyx = %10.6f, dAyy = %10.6f \n", dGpt[0], dGpt[1], dGpt[2], dGpt[3]);
		// print4x4(M);
		// print4x4(Minv);
	}

	tGpt1[0][2] = (gx2 - tGpt1[0][0] * gx1 - tGpt1[0][1] * gy1) / g0;
	tGpt1[1][2] = (gy2 - tGpt1[1][0] * gx1 - tGpt1[1][1] * gy1) / g0;

	tGpt1[2][0] = tGpt1[2][1] = 0.0;
	tGpt1[2][2] = 1.0;

	/* printf("datr a11 = %f, a12 = %f, a21 = %f, a22 = %f, b1 = %f b2 = %f\n", a1[0][0], a1[0][1], a1[1][0], a1[1][1], b1[0], b1[1]); */

	/* update of GAT components */
	// print3x3(gpt);
	// print3x3(tGpt1);
	multiply3x3(tGpt1, gpt, tGpt2);
	copyGpt(tGpt2, gpt);
}

void npptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
		int g_ang2[ROW][COL], double g_can2[ROW][COL],
		double gwt[ROW][COL], double gpt[3][3], double dnn) {
	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2;
	double* g = (double*) malloc (G_NUM * sizeof(double));
	double tv, t0, tx2, ty2;
	double det, U11, U12, U21, U22, Uinv11, Uinv12, Uinv21, Uinv22, v1, v2;
	double dx1, dx2, dy1, dy2;
	double tIn12, tIn11, gIn12x1x1, gIn12x1y1, gIn12y1y1, gIn11x1, gIn11y1, gIn12x1, gIn12y1;
	double tGpt1[3][3], tGpt2[3][3];
	double gIn11, gIn12, gIn22;

	/* Gaussian weigthed mean values */
	gIn12x1x1 = gIn12x1y1 = gIn12y1y1 = gIn11x1 = gIn11y1 = gIn12x1 = gIn12y1 = 0.0;
	gIn11 = gIn12 = gIn22 = 0.0;
	gx1 = gy1 = 0.0;
	for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
		dy1 = y1 - CY;
		for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
			dx1 = x1 - CX;
			t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
			for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
				dy2 = y2 - CY;
				for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
					dx2 = x2 - CX;
					if (g_ang1[y1][x1] == g_ang2[y2][x2] && g_ang1[y1][x1] != -1) {
						tv   = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can1[y1][x1] * g_can2[y2][x2];
						t0    += tv;
						tx2   += tv * dx2;
						ty2   += tv * dy2;
						gIn22 += tv * (dx2 * dx2 + dy2 * dy2);
					}
				}
			}
			g0        += t0;
			tIn12      =  dx1 * tx2 + dy1 * ty2;
			tIn11      = (dx1 * dx1 + dy1 * dy1) * t0;
			gIn12x1x1 += tIn12 * dx1 * dx1;
			gIn12x1y1 += tIn12 * dx1 * dy1;
			gIn12y1y1 += tIn12 * dy1 * dy1;
			gIn11x1   += tIn11 * dx1;
			gIn11y1   += tIn11 * dy1;
			gIn12x1   += tIn12 * dx1;
			gIn12y1   += tIn12 * dy1;
			gIn11     += (dx1 * dx1 + dy1 * dy1) * t0;
			gIn12     += dx1 * tx2 + dy1 * ty2;
			gx1       += t0 * dx1;
			gy1       += t0 * dy1;
		}
	}

	// printf("gx1x1 = %f gx1x2 = %f | gy1y1 = %f gy1y2 = %f\n", gx1x1/g0, gx1x2/g0, gy1y1/g0, gy1y2/g0);

	if (g0 == 0.0) {
		printf("PPT calculation failure by zero sum!!!\n");
		return;
	}

	U11 =       gIn12x1x1;
	U12 = U21 = gIn12x1y1;
	U22 =       gIn12y1y1;

	/* U^{-1} */
	det = U11 * U22 - U21 * U12;
	if (fabs(det) < EPS) {
		printf("PPT calculation failure by zero det of V for Both side !!! det = %f\n", det);
		return;
	}
	Uinv11 =  U22 / det;
	Uinv12 = -U12 / det;
	Uinv21 = -U21 / det;
	Uinv22 =  U11 / det;

	/* V = bar(<x_1 x_1> x_1) - bar(<x_1 x_2> x_1) */
	v1 = gIn11x1 - gIn12x1 - dnn * dnn * gx1 / 2;
	v2 = gIn11y1 - gIn12y1 - dnn * dnn * gy1 / 2;

	tGpt1[0][0] = 1.0; tGpt1[0][1] = 0.0; tGpt1[0][2] = 0.0;
	tGpt1[1][0] = 0.0; tGpt1[1][1] = 1.0; tGpt1[1][2] = 0.0;

	tGpt1[2][0] = Uinv11 * v1 + Uinv12 * v2;
	tGpt1[2][1] = Uinv21 * v1 + Uinv22 * v2;

	tGpt1[2][2] = 1.0;

	// printf("c1 = %f  c2 = %f \n", tGpt1[2][0], tGpt1[2][1]);
	// printf("gIn11 = %f  gIn12 = %f  gIn22 = %f \n", gIn11, gIn12, gIn22);

	/* update of GAT components */
	multiply3x3(tGpt1, gpt, tGpt2);
	copyGpt(tGpt2, gpt);
}

void sgptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
		int g_ang2[ROW][COL], double g_can2[ROW][COL],
		double gwt[ROW][COL], double gpt[3][3]) {
	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2;
	// double gx1, gy1;
	// double g0, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3;
	// double gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2;
	double* g = (double*) malloc (G_NUM * sizeof(double));
	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double denom;
	double dx1, dx2, dy1, dy2;
	double tGpt1[3][3], tGpt2[3][3];

	int count = 0;
	/* Gaussian weigthed mean values */
	g0 = gx2 = gy2 = 0.0;
	gx1p1 = gx1p2 = gx1p3 = gx1p4 = gy1p1 = gy1p2 = gy1p3 = gy1p4 = 0.0;
	gx1p1y1p1 = gx1p2y1p1 = gx1p3y1p1 = gx1p1y1p2 = gx1p2y1p2 = gx1p1y1p3 = 0.0;
	gx1x2 = gx1y2 = gy1x2 = gy1y2 = 0.0;
	gx1p2x2 = gx1y1y2 = gx1y1x2 = gy1p2y2 = 0.0;
	for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
		dy1 = y1 - CY;
		for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
			dx1 = x1 - CX;
			t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
			for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
				dy2 = y2 - CY;
				for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
					dx2 = x2 - CX;
					if (g_ang1[y1][x1] == g_ang2[y2][x2] && g_ang1[y1][x1] != -1) {

						tv   = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can1[y1][x1] * g_can2[y2][x2];
						t0    += tv;
						tx2   += tv * dx2;
						ty2   += tv * dy2;

					}
				}
			}
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


	if (g0 == 0.0) {
		printf("GAT calculation failure by zero sum!!!\n");
		return;
	}

	double Mat8[8][8], Para[8], Const[8], iMat8[8][8], eMat8[8][8];

	Mat8[0][0] = gx1p2;     Mat8[0][1] = gx1p1y1p1; Mat8[0][2] =  0.0;       Mat8[0][3] =  0.0;
	Mat8[0][4] = gx1p1;     Mat8[0][5] = 0.0;       Mat8[0][6] = -gx1p3;     Mat8[0][7] = -gx1p2y1p1;

	Mat8[1][0] = gx1p1y1p1; Mat8[1][1] = gy1p2;     Mat8[1][2] =  0.0;       Mat8[1][3] =  0.0;
	Mat8[1][4] = gy1p1;     Mat8[1][5] = 0.0;       Mat8[1][6] = -gx1p2y1p1; Mat8[1][7] = -gx1p1y1p2;

	Mat8[2][0] = 0.0;       Mat8[2][1] = 0.0;       Mat8[2][2] =  gx1p2;     Mat8[2][3] =  gx1p1y1p1;
	Mat8[2][4] = 0.0;       Mat8[2][5] = gx1p1;     Mat8[2][6] = -gx1p2y1p1; Mat8[2][7] = -gx1p1y1p2;

	Mat8[3][0] = 0.0;       Mat8[3][1] = 0.0;       Mat8[3][2] =  gx1p1y1p1; Mat8[3][3] =  gy1p2;
	Mat8[3][4] = 0.0;       Mat8[3][5] = gy1p1;     Mat8[3][6] = -gx1p1y1p2; Mat8[3][7] = -gy1p3;

	Mat8[4][0] = gx1p1;     Mat8[4][1] = gy1p1;     Mat8[4][2] =  0.0;       Mat8[4][3] =  0.0;
	Mat8[4][4] = g0;        Mat8[4][5] = 0.0;       Mat8[4][6] = -gx1p2;     Mat8[4][7] = -gx1p1y1p1;

	Mat8[5][0] = 0.0;       Mat8[5][1] = 0.0;       Mat8[5][2] =  gx1p1;     Mat8[5][3] =  gy1p1;
	Mat8[5][4] = 0.0;       Mat8[5][5] = g0;        Mat8[5][6] = -gx1p1y1p1; Mat8[5][7] = -gy1p2;

	Mat8[6][0] = gx1p3;     Mat8[6][1] = gx1p2y1p1; Mat8[6][2] =  gx1p2y1p1; Mat8[6][3] =  gx1p1y1p2;
	Mat8[6][4] = gx1p2;     Mat8[6][5] = gx1p1y1p1; Mat8[6][6] = -(gx1p4 + gx1p2y1p2);     Mat8[6][7] = -(gx1p3y1p1 + gx1p1y1p3);

	Mat8[7][0] = gx1p2y1p1; Mat8[7][1] = gx1p1y1p2; Mat8[7][2] =  gx1p1y1p2; Mat8[7][3] =  gy1p3;
	Mat8[7][4] = gx1p1y1p1; Mat8[7][5] = gy1p2;     Mat8[7][6] = -(gx1p3y1p1 + gx1p1y1p3); Mat8[7][7] = -(gx1p2y1p2 + gy1p4);

	Const[0] = gx1x2;
	Const[1] = gy1x2;
	Const[2] = gx1y2;
	Const[3] = gy1y2;
	Const[4] = gx2;
	Const[5] = gy2;
	Const[6] = (gx1p2x2 + gx1y1y2);
	Const[7] = (gx1y1x2 + gy1p2y2);


	// I need a inverse matrix
	inverse8x8(Mat8, iMat8);
	multiplyVect8x8(iMat8, Const, Para);

	// multiply8x8(Mat8, iMat8, eMat8);
	// print8x8(Mat8);
	// print8x8(iMat8);
	// print8x8(eMat8);

	/* update of GAT components */

	tGpt2[0][0] = Para[0]; tGpt2[0][1] = Para[1]; tGpt2[1][0] = Para[2]; tGpt2[1][1] = Para[3];
	tGpt2[0][2] = Para[4]; tGpt2[1][2] = Para[5];
	tGpt2[2][0] = Para[6]; tGpt2[2][1] = Para[7];
	tGpt2[2][2] = 1.0;
	//print3x3(tGpt2);
	multiply3x3(tGpt2, gpt, tGpt1);
	copyGpt(tGpt1, gpt);

	int i, j;
	for (i = 0 ; i < 3 ; ++i) {
		for (j = 0 ; j < 3 ; ++j) {
			gpt[i][j] /= gpt[2][2];
		}
	}

	// print3x3(gpt);
}

void nsgptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL],
		int g_ang2[ROW][COL], double g_can2[ROW][COL],
		double gwt[ROW][COL], double gpt[3][3], double dnn) {
	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2;
	// double gx1, gy1;
	// double g0, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3;
	// double gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2;
	double* g = (double*) malloc (G_NUM * sizeof(double));
	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double denom;
	double dx1, dx2, dy1, dy2;
	double tGpt1[3][3], tGpt2[3][3];

	int count = 0;
	/* Gaussian weigthed mean values */
	g0 = gx2 = gy2 = 0.0;
	gx1p1 = gx1p2 = gx1p3 = gx1p4 = gy1p1 = gy1p2 = gy1p3 = gy1p4 = 0.0;
	gx1p1y1p1 = gx1p2y1p1 = gx1p3y1p1 = gx1p1y1p2 = gx1p2y1p2 = gx1p1y1p3 = 0.0;
	gx1x2 = gx1y2 = gy1x2 = gy1y2 = 0.0;
	gx1p2x2 = gx1y1y2 = gx1y1x2 = gy1p2y2 = 0.0;
	for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
		dy1 = y1 - CY;
		for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
			dx1 = x1 - CX;
			// if (g_ang1[y1][x1] == -1) continue;
			t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
			for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
				dy2 = y2 - CY;
				for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
					dx2 = x2 - CX;
					if (g_ang1[y1][x1] == g_ang2[y2][x2] && g_ang1[y1][x1] != -1) {
						tv   = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can1[y1][x1] * g_can2[y2][x2];
						t0    += tv;
						tx2   += tv * dx2;
						ty2   += tv * dy2;
						count++;
					}
				}
			}
			// if(t0 != 0.0)
			//     printf("(%d, %d), t0 = %f \n", y1, x1, t0);

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


	if (g0 == 0.0) {
		printf("GAT calculation failure by zero sum!!!\n");
		return;
	}

	// printf("g0 = %f\n", g0);

	double Mat8[8][8], Para[8], Const[8], iMat8[8][8], eMat8[8][8];

	Mat8[0][0] = gx1p2;     Mat8[0][1] = gx1p1y1p1; Mat8[0][2] =  0.0;       Mat8[0][3] =  0.0;
	Mat8[0][4] = gx1p1;     Mat8[0][5] = 0.0;       Mat8[0][6] = -gx1p3;     Mat8[0][7] = -gx1p2y1p1;

	Mat8[1][0] = gx1p1y1p1; Mat8[1][1] = gy1p2;     Mat8[1][2] =  0.0;       Mat8[1][3] =  0.0;
	Mat8[1][4] = gy1p1;     Mat8[1][5] = 0.0;       Mat8[1][6] = -gx1p2y1p1; Mat8[1][7] = -gx1p1y1p2;

	Mat8[2][0] = 0.0;       Mat8[2][1] = 0.0;       Mat8[2][2] =  gx1p2;     Mat8[2][3] =  gx1p1y1p1;
	Mat8[2][4] = 0.0;       Mat8[2][5] = gx1p1;     Mat8[2][6] = -gx1p2y1p1; Mat8[2][7] = -gx1p1y1p2;

	Mat8[3][0] = 0.0;       Mat8[3][1] = 0.0;       Mat8[3][2] =  gx1p1y1p1; Mat8[3][3] =  gy1p2;
	Mat8[3][4] = 0.0;       Mat8[3][5] = gy1p1;     Mat8[3][6] = -gx1p1y1p2; Mat8[3][7] = -gy1p3;

	Mat8[4][0] = gx1p1;     Mat8[4][1] = gy1p1;     Mat8[4][2] =  0.0;       Mat8[4][3] =  0.0;
	Mat8[4][4] = g0;        Mat8[4][5] = 0.0;       Mat8[4][6] = -gx1p2;     Mat8[4][7] = -gx1p1y1p1;

	Mat8[5][0] = 0.0;       Mat8[5][1] = 0.0;       Mat8[5][2] =  gx1p1;     Mat8[5][3] =  gy1p1;
	Mat8[5][4] = 0.0;       Mat8[5][5] = g0;        Mat8[5][6] = -gx1p1y1p1; Mat8[5][7] = -gy1p2;

	Mat8[6][0] = gx1p3;     Mat8[6][1] = gx1p2y1p1; Mat8[6][2] =  gx1p2y1p1; Mat8[6][3] =  gx1p1y1p2;
	Mat8[6][4] = gx1p2;     Mat8[6][5] = gx1p1y1p1; Mat8[6][6] = -(gx1p4 + gx1p2y1p2);     Mat8[6][7] = -(gx1p3y1p1 + gx1p1y1p3);

	Mat8[7][0] = gx1p2y1p1; Mat8[7][1] = gx1p1y1p2; Mat8[7][2] =  gx1p1y1p2; Mat8[7][3] =  gy1p3;
	Mat8[7][4] = gx1p1y1p1; Mat8[7][5] = gy1p2;     Mat8[7][6] = -(gx1p3y1p1 + gx1p1y1p3); Mat8[7][7] = -(gx1p2y1p2 + gy1p4);

	Const[0] = gx1x2;
	Const[1] = gy1x2;
	Const[2] = gx1y2;
	Const[3] = gy1y2;
	Const[4] = gx2;
	Const[5] = gy2;
	Const[6] = (gx1p2x2 + gx1y1y2);
	Const[7] = (gx1y1x2 + gy1p2y2);


	// inverse matrix
	inverse8x8(Mat8, iMat8);
	multiplyVect8x8(iMat8, Const, Para);


	// Newton-Raphson here initial value is Para[8]

	int iter;
	double dGpt[8], tempC8[8];
	double W2, detA;
	double T00, T01, T10, T11;
	W2   = dnn * dnn;
	for (iter = 0 ; iter < MAXNR ; ++iter) {
		detA = Para[0] * Para[3] - Para[1] * Para[2];

		T00 =  Para[3] / detA;
		T01 = -Para[1] / detA;
		T10 = -Para[2] / detA;
		T11 =  Para[0] / detA;

		int i, j;
		// submit Mat8 to eMat8 and update eMat8
		for (i = 0 ; i < 8 ; ++i) {
			for (j = 0 ; j < 8 ; ++j) {
				eMat8[i][j] = Mat8[i][j];
			}
		}

		eMat8[0][0] += 0.5 * W2 * T00 * T00 * g0;
		eMat8[0][1] += 0.5 * W2 * T00 * T10 * g0;
		eMat8[0][2] += 0.5 * W2 * T01 * T00 * g0;
		eMat8[0][3] += 0.5 * W2 * T01 * T10 * g0;

		eMat8[1][0] += 0.5 * W2 * T10 * T00 * g0;
		eMat8[1][1] += 0.5 * W2 * T10 * T11 * g0;
		eMat8[1][2] += 0.5 * W2 * T11 * T00 * g0;
		eMat8[1][3] += 0.5 * W2 * T11 * T11 * g0;

		eMat8[2][0] += 0.5 * W2 * T00 * T01 * g0;
		eMat8[2][1] += 0.5 * W2 * T00 * T11 * g0;
		eMat8[2][2] += 0.5 * W2 * T01 * T01 * g0;
		eMat8[2][3] += 0.5 * W2 * T01 * T11 * g0;

		eMat8[3][0] += 0.5 * W2 * T10 * T01 * g0;
		eMat8[3][1] += 0.5 * W2 * T10 * T11 * g0;
		eMat8[3][2] += 0.5 * W2 * T11 * T01 * g0;
		eMat8[3][3] += 0.5 * W2 * T11 * T11 * g0;

		eMat8[6][6] += 0.25 * W2 * gx1p2;
		eMat8[6][7] += 0.25 * W2 * gx1p1y1p1;

		eMat8[7][6] += 0.25 * W2 * gx1p1y1p1;
		eMat8[7][7] += 0.25 * W2 * gy1p2;

		// Const
		Const[0] = gx1x2 + 0.5 * W2 * T00 * g0;
		Const[1] = gy1x2 + 0.5 * W2 * T10 * g0;
		Const[2] = gx1y2 + 0.5 * W2 * T01 * g0;
		Const[3] = gy1y2 + 0.5 * W2 * T11 * g0;
		Const[4] = gx2;
		Const[5] = gy2;
		Const[6] = gx1p2x2 + gx1y1y2 + 0.5 * W2 * ( gx1p1 - 0.5 * Para[6] * gx1p2 - 0.5 * Para[7] * gx1p1y1p1 );
		Const[7] = gx1y1x2 + gy1p2y2 + 0.5 * W2 * ( gy1p1 - 0.5 * Para[6] * gx1p1y1p1 - 0.5 * Para[7] * gy1p2 );

		// calculate iMat8
		inverse8x8(eMat8, iMat8);
		// tempC8 = Mat8 * Para;
		multiplyVect8x8(Mat8, Para, tempC8);
		// tempC8 = Const -tempC8;
		for (i = 0 ; i < 8 ; ++i) tempC8[i] = Const[i] - tempC8[i];
		// dGpt = iMat8 * tempC8;
		multiplyVect8x8(iMat8, tempC8, dGpt);

		double zeroConfirm = 0.0;
		for (i = 0 ; i < 8 ; ++i) {
			zeroConfirm += dGpt[i] * dGpt[i];
			Para[i] += dGpt[i];
		}

		if (zeroConfirm < EPS2) break;

		printf("Zero confirm value : %f \n", zeroConfirm);

	}

	tGpt2[0][0] = Para[0]; tGpt2[0][1] = Para[1]; tGpt2[1][0] = Para[2]; tGpt2[1][1] = Para[3];
	tGpt2[0][2] = Para[4]; tGpt2[1][2] = Para[5];
	tGpt2[2][0] = Para[6]; tGpt2[2][1] = Para[7];
	tGpt2[2][2] = 1.0;

	/* update of GAT components */
	//print3x3(tGpt2);
	multiply3x3(tGpt2, gpt, tGpt1);
	copyGpt(tGpt1, gpt);

	int i, j;
	for (i = 0 ; i < 3 ; ++i) {
		for (j = 0 ; j < 3 ; ++j) {
			gpt[i][j] /= gpt[2][2];
		}
	}

	// print3x3(gpt);
	printf("Number of matched points is %d \n", count);
}

void nsgptcorSpHOG5x5(int g_ang1[ROW][COL], char sHoG1[ROW - 4][COL - 4], double g_can1[ROW][COL],
		int g_ang2[ROW][COL], char sHoG2[ROW - 4][COL - 4], double g_can2[ROW][COL],
		double gwt[ROW][COL], double gpt[3][3], double dnn) {
	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2, nDir;
	char HoG1[8], HoG2[8];
	// double gx1, gy1;
	// double g0, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3;
	// double gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2;
	double* g = (double*) malloc (G_NUM * sizeof(double));
	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double denom;
	double dx1, dx2, dy1, dy2;
	double tGpt1[3][3], tGpt2[3][3];

	int margin = 2;
	int count = 0;
	/* Gaussian weigthed mean values */
	g0 = gx2 = gy2 = 0.0;
	gx1p1 = gx1p2 = gx1p3 = gx1p4 = gy1p1 = gy1p2 = gy1p3 = gy1p4 = 0.0;
	gx1p1y1p1 = gx1p2y1p1 = gx1p3y1p1 = gx1p1y1p2 = gx1p2y1p2 = gx1p1y1p3 = 0.0;
	gx1x2 = gx1y2 = gy1x2 = gy1y2 = 0.0;
	gx1p2x2 = gx1y1y2 = gx1y1x2 = gy1p2y2 = 0.0;
	for (y1 = margin ; y1 < ROW - margin ; y1++) {
		dy1 = y1 - CY;
		for (x1 = margin ; x1 < COL - margin ; x1++) {
			dx1 = x1 - CX;
			if (sHoG1[y1 - margin][x1 - margin] == -1) continue;
			t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
			for (y2 = margin ; y2 < ROW - margin ; y2++) {
				dy2 = y2 - CY;
				for (x2 = margin ; x2 < COL - margin ; x2++) {
					dx2 = x2 - CX;

					if (sHoG2[y2 - margin][x2 - margin] == -1) continue;

					if (sHoG1[y1 - margin][x1 - margin] == sHoG2[y2 - margin][x2 - margin]) {

						tv   = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can1[y1][x1] * g_can2[y2][x2];
						t0    += tv;
						tx2   += tv * dx2;
						ty2   += tv * dy2;
						count++;

					}
				}
			}
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


	if (g0 == 0.0) {
		printf("GAT calculation failure by zero sum!!!\n");
		return;
	}

	double Mat8[8][8], Para[8], Const[8], iMat8[8][8], eMat8[8][8];

	Mat8[0][0] = gx1p2;     Mat8[0][1] = gx1p1y1p1; Mat8[0][2] =  0.0;       Mat8[0][3] =  0.0;
	Mat8[0][4] = gx1p1;     Mat8[0][5] = 0.0;       Mat8[0][6] = -gx1p3;     Mat8[0][7] = -gx1p2y1p1;

	Mat8[1][0] = gx1p1y1p1; Mat8[1][1] = gy1p2;     Mat8[1][2] =  0.0;       Mat8[1][3] =  0.0;
	Mat8[1][4] = gy1p1;     Mat8[1][5] = 0.0;       Mat8[1][6] = -gx1p2y1p1; Mat8[1][7] = -gx1p1y1p2;

	Mat8[2][0] = 0.0;       Mat8[2][1] = 0.0;       Mat8[2][2] =  gx1p2;     Mat8[2][3] =  gx1p1y1p1;
	Mat8[2][4] = 0.0;       Mat8[2][5] = gx1p1;     Mat8[2][6] = -gx1p2y1p1; Mat8[2][7] = -gx1p1y1p2;

	Mat8[3][0] = 0.0;       Mat8[3][1] = 0.0;       Mat8[3][2] =  gx1p1y1p1; Mat8[3][3] =  gy1p2;
	Mat8[3][4] = 0.0;       Mat8[3][5] = gy1p1;     Mat8[3][6] = -gx1p1y1p2; Mat8[3][7] = -gy1p3;

	Mat8[4][0] = gx1p1;     Mat8[4][1] = gy1p1;     Mat8[4][2] =  0.0;       Mat8[4][3] =  0.0;
	Mat8[4][4] = g0;        Mat8[4][5] = 0.0;       Mat8[4][6] = -gx1p2;     Mat8[4][7] = -gx1p1y1p1;

	Mat8[5][0] = 0.0;       Mat8[5][1] = 0.0;       Mat8[5][2] =  gx1p1;     Mat8[5][3] =  gy1p1;
	Mat8[5][4] = 0.0;       Mat8[5][5] = g0;        Mat8[5][6] = -gx1p1y1p1; Mat8[5][7] = -gy1p2;

	Mat8[6][0] = gx1p3;     Mat8[6][1] = gx1p2y1p1; Mat8[6][2] =  gx1p2y1p1; Mat8[6][3] =  gx1p1y1p2;
	Mat8[6][4] = gx1p2;     Mat8[6][5] = gx1p1y1p1; Mat8[6][6] = -(gx1p4 + gx1p2y1p2);     Mat8[6][7] = -(gx1p3y1p1 + gx1p1y1p3);

	Mat8[7][0] = gx1p2y1p1; Mat8[7][1] = gx1p1y1p2; Mat8[7][2] =  gx1p1y1p2; Mat8[7][3] =  gy1p3;
	Mat8[7][4] = gx1p1y1p1; Mat8[7][5] = gy1p2;     Mat8[7][6] = -(gx1p3y1p1 + gx1p1y1p3); Mat8[7][7] = -(gx1p2y1p2 + gy1p4);

	Const[0] = gx1x2;
	Const[1] = gy1x2;
	Const[2] = gx1y2;
	Const[3] = gy1y2;
	Const[4] = gx2;
	Const[5] = gy2;
	Const[6] = (gx1p2x2 + gx1y1y2);
	Const[7] = (gx1y1x2 + gy1p2y2);


	// inverse matrix
	inverse8x8(Mat8, iMat8);
	multiplyVect8x8(iMat8, Const, Para);


	// Newton-Raphson here initial value is Para[8]

	int iter;
	double dGpt[8], tempC8[8];
	double W2, detA;
	double T00, T01, T10, T11;
	W2   = dnn * dnn;
	for (iter = 0 ; iter < MAXNR ; ++iter) {
		detA = Para[0] * Para[3] - Para[1] * Para[2];

		T00 =  Para[3] / detA;
		T01 = -Para[1] / detA;
		T10 = -Para[2] / detA;
		T11 =  Para[0] / detA;

		int i, j;
		// submit Mat8 to eMat8 and update eMat8
		for (i = 0 ; i < 8 ; ++i) {
			for (j = 0 ; j < 8 ; ++j) {
				eMat8[i][j] = Mat8[i][j];
			}
		}

		eMat8[0][0] += 0.5 * W2 * T00 * T00 * g0;
		eMat8[0][1] += 0.5 * W2 * T00 * T10 * g0;
		eMat8[0][2] += 0.5 * W2 * T01 * T00 * g0;
		eMat8[0][3] += 0.5 * W2 * T01 * T10 * g0;

		eMat8[1][0] += 0.5 * W2 * T10 * T00 * g0;
		eMat8[1][1] += 0.5 * W2 * T10 * T11 * g0;
		eMat8[1][2] += 0.5 * W2 * T11 * T00 * g0;
		eMat8[1][3] += 0.5 * W2 * T11 * T11 * g0;

		eMat8[2][0] += 0.5 * W2 * T00 * T01 * g0;
		eMat8[2][1] += 0.5 * W2 * T00 * T11 * g0;
		eMat8[2][2] += 0.5 * W2 * T01 * T01 * g0;
		eMat8[2][3] += 0.5 * W2 * T01 * T11 * g0;

		eMat8[3][0] += 0.5 * W2 * T10 * T01 * g0;
		eMat8[3][1] += 0.5 * W2 * T10 * T11 * g0;
		eMat8[3][2] += 0.5 * W2 * T11 * T01 * g0;
		eMat8[3][3] += 0.5 * W2 * T11 * T11 * g0;

		eMat8[6][6] += 0.25 * W2 * gx1p2;
		eMat8[6][7] += 0.25 * W2 * gx1p1y1p1;

		eMat8[7][6] += 0.25 * W2 * gx1p1y1p1;
		eMat8[7][7] += 0.25 * W2 * gy1p2;

		// Const
		Const[0] = gx1x2 + 0.5 * W2 * T00 * g0;
		Const[1] = gy1x2 + 0.5 * W2 * T10 * g0;
		Const[2] = gx1y2 + 0.5 * W2 * T01 * g0;
		Const[3] = gy1y2 + 0.5 * W2 * T11 * g0;
		Const[4] = gx2;
		Const[5] = gy2;
		Const[6] = gx1p2x2 + gx1y1y2 + 0.5 * W2 * ( gx1p1 - 0.5 * Para[6] * gx1p2 - 0.5 * Para[7] * gx1p1y1p1 );
		Const[7] = gx1y1x2 + gy1p2y2 + 0.5 * W2 * ( gy1p1 - 0.5 * Para[6] * gx1p1y1p1 - 0.5 * Para[7] * gy1p2 );

		// calculate iMat8
		inverse8x8(eMat8, iMat8);
		// tempC8 = Mat8 * Para;
		multiplyVect8x8(Mat8, Para, tempC8);
		// tempC8 = Const -tempC8;
		for (i = 0 ; i < 8 ; ++i) tempC8[i] = Const[i] - tempC8[i];
		// dGpt = iMat8 * tempC8;
		multiplyVect8x8(iMat8, tempC8, dGpt);

		double zeroConfirm = 0.0;
		for (i = 0 ; i < 8 ; ++i) {
			zeroConfirm += dGpt[i] * dGpt[i];
			Para[i] += dGpt[i];
		}

		if (zeroConfirm < EPS2) break;

		// printf("Zero confirm value : %f \n", zeroConfirm);

	}

	tGpt2[0][0] = Para[0]; tGpt2[0][1] = Para[1]; tGpt2[1][0] = Para[2]; tGpt2[1][1] = Para[3];
	tGpt2[0][2] = Para[4]; tGpt2[1][2] = Para[5];
	tGpt2[2][0] = Para[6]; tGpt2[2][1] = Para[7];
	tGpt2[2][2] = 1.0;

	/* update of GAT components */
	//print3x3(tGpt2);
	multiply3x3(tGpt2, gpt, tGpt1);
	copyGpt(tGpt1, gpt);

	int i, j;
	for (i = 0 ; i < 3 ; ++i) {
		for (j = 0 ; j < 3 ; ++j) {
			gpt[i][j] /= gpt[2][2];
		}
	}

	// print3x3(gpt);
	printf("Number of matched points is %d \n", count);
}
