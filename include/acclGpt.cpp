#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

#include "parameter.h"
#include "utility.h"
#include "stdGpt.h"
#include "acclGpt.h"
#include "acclGpt_cuda.h"

using namespace std;

void winTbl(int g_ang2[ROW][COL], double D[ROW][COL * 8], char *fn);
void winTbl64(char sHoG2[ROW - 4][COL - 4], double D2[ROW - 4][(COL - 4) * 64], char *fn);
void searchTbl(int row, int col, char *);
void loadTbls(double D[ROW][COL * 8], double D2[ROW - 4][(COL - 4) * 64], double ndis[(2 * ROW - 1) * (2 * COL - 1)], int coor[(2 * ROW - 1) * (2 * COL - 1)][2]);
void makeTemp(int g_ang2[ROW][COL], double g_can2[ROW][COL], double gt[ROW][COL], double H[ROW][COL * 162], char *);
void makeTemp64(char sHoG2[ROW - 4][COL - 4], double g_can2[ROW][COL], double gwt[ROW][COL], double H[ROW - 4][(COL - 4) * 6 * 64 * 3], char *);
void makeTemp64_far(char sHoG2[ROW - 4][COL - 4], double g_can2[ROW][COL], double gwt[ROW][COL], double H[ROW - 4][(COL - 4) * 6 * 64 * 3], char *);
void loadTemp(double H[ROW][COL * 162]);
void loadTemp64(double H[ROW - 4][(COL - 4) * 6 * 64 * 3]);
void loadTemp64_far(double H[ROW - 4][(COL - 4) * 6 * 64 * 3]);

double fwinpat(int g_ang1[ROW][COL], int g_ang2[ROW][COL], double D[ROW][COL * 8], double ndis[(2 * ROW - 1) * (2 * COL - 1)], int coor[(2 * ROW - 1) * (2 * COL - 1)][2]);
double fsHoGpat(char sHoG1[ROW - 4][COL - 4], char sHoG2[ROW - 4][COL - 4], double D[ROW - 4][(COL - 4) * 64], double ndis[(2 * ROW - 1) * (2 * COL - 1)], int coor[(2 * ROW - 1) * (2 * COL - 1)][2]);

void fgatcor(int g_ang1[ROW][COL], double g_can1[ROW][COL], double gpt[3][3], double dnn, double Ht[ROW][COL * 27], char *);
void fngatcor(int g_ang1[ROW][COL], double g_can1[ROW][COL], double gpt[3][3], double dnn, double Ht[ROW][COL * 27], char *);
void fpptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL], double gpt[3][3], double dnn, double Ht[ROW][COL * 27], char *);
void fnpptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL], double gpt[3][3], double dnn, double Ht[ROW][COL * 27], char *);
void fsgptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL], double gpt[3][3], double dnn, double Ht[ROW][COL * 27], char *);
void fnsgptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL], double gpt[3][3], double dnn, double H[ROW][COL * 162], double Ht[ROW][COL * 27]);
void fnsgptcorSpHOG5x5(int g_ang1[ROW][COL], char sHoG1[ROW - 4][COL - 4], double g_can1[ROW][COL],
                     double gpt[3][3], double dnn, double H[ROW - 4][(COL - 4) * 6 * 64 * 3], double Ht[ROW - 4][(COL - 4) * 64 * 3]);
void fnsgptcorSpHOG5x5_far(int g_ang1[ROW][COL], char sHoG1[ROW - 4][COL - 4], double g_can1[ROW][COL],
                     double gpt[3][3], double dnn, double H[ROW - 4][(COL - 4) * 6 * 64 * 3], double Ht[ROW - 4][(COL - 4) * 64 * 3]);

/****************************************************************/
void winTbl(int g_ang2[ROW][COL], double D[ROW][COL * 8], char *fn) {
    char fnt[128];
    int x2, y2, tx2, ty2, s;
    // double D[ROW][COL * 8];
    double minInit, min, delta;
    
    sprintf(fnt, "%s_DTbl", fn);
    
    minInit = (ROW - 2 * MARGINE) * (ROW - 2 * MARGINE) + (COL - 2 * MARGINE) * (COL - 2 * MARGINE);
    for (s = 0 ; s < 8 ; s++) {
        for (y2 = MARGINE ; y2 < ROW - MARGINE; y2++) {
            for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
                min = minInit;
                for (ty2 = MARGINE ; ty2 < ROW - MARGINE ; ty2++) {
                    for (tx2 = MARGINE ; tx2 < COL - MARGINE ; tx2++) {
                        if (g_ang2[ty2][tx2] == s) {
                            delta = (y2 - ty2) * (y2 - ty2) + (x2 - tx2) * (x2 - tx2);
                            if (delta < min) min = delta;
                        }
                    }
                }
                D[y2][x2 + s * COL] = sqrt(min);
            }
        }
    }
    
    /* Make file pointer */
    FILE *fp;
    if((fp = fopen(fnt, "wb")) == NULL ) {
        printf("\nCannot open the file! \n");
        exit(EXIT_FAILURE);
    }
    fwrite(D,  sizeof(double), COL * ROW * 8, fp);
    fclose(fp);
    
    /*
    sprintf(fnt, "%s.csv", fnt);
    if((fp = fopen(fnt, "w")) == NULL ) {
        printf("\nCannot open the file! \n");
        exit(EXIT_FAILURE);
    }
    for (y2 = 0 ; y2 < ROW ; y2++) {
        for (x2 = 0 ; x2 < COL ; x2++) {
            fprintf(fp, "%d,", g_ang2[y2][x2]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    for (y2 = 0 ; y2 < ROW ; y2++) {
        for (x2 = 0 ; x2 < 8 * COL ; x2++) {
            fprintf(fp, "%f,", D[y2][x2]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp); */
    printf("Finish \n");
}

void winTbl64(char sHoG2[ROW - 4][COL - 4], double D2[ROW - 4][(COL - 4) * 64], char *fn) {
	char fnt[128];
	int x2, y2, tx2, ty2, s, tempS;
	// double D[ROW - 4][(COL - 4) * 64];
	double minInit, min, delta;

	sprintf(fnt, "%s_DTbl64", fn);

	minInit = (ROW - 2 * MARGINE - 4) * (ROW - 2 * MARGINE - 4) + (COL - 2 * MARGINE - 4) * (COL - 2 * MARGINE - 4);

	for (s = 0 ; s < 64 ; s++) {
		printf("s = %d \n", s);
		for (y2 = 0 ; y2 < ROW - 4; y2++) {
			for (x2 = 0 ; x2 < COL - 4 ; x2++) {
				min = minInit;
				for (ty2 = 0 ; ty2 < ROW - 4 ; ty2++) {
					for (tx2 = 0 ; tx2 < COL - 4 ; tx2++) {
						tempS = sHoG2Idx(sHoG2[ty2][tx2]);
						if (tempS == s) {
							delta = (y2 - ty2) * (y2 - ty2) + (x2 - tx2) * (x2 - tx2);
							if (delta < min) min = delta;
						}
					}
				}
				D2[y2][x2 + s * (COL - 4)] = sqrt(min);
			}
		}
	}
	/*
	for (y2 = 40 ; y2 < 50 ; y2++)
		printf("H[y2][0] = %d | H[y2][1] = %d \n", sHoG2Idx(sHoG2[y2][0]), sHoG2Idx(sHoG2[y2][1]));
	for (y2 = 40 ; y2 < 50 ; y2++)
		printf("D[y2][0] = %f \n", D2[y2][0]);
		*/
	/* Make file pointer */
	FILE *fp;
	if((fp = fopen(fnt, "wb")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	fwrite(D2,  sizeof(double), (ROW - 4) * (COL - 4) * 64, fp);
	fclose(fp);

	/*
	sprintf(fnt, "%s.csv", fnt);
	if((fp = fopen(fnt, "w")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	for (y2 = 0 ; y2 < ROW - 4 ; y2++) {
		for (x2 = 0 ; x2 < COL - 4 ; x2++) {
			fprintf(fp, "%d,", sHoG2Idx(sHoG2[y2][x2]));
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	for (y2 = 0 ; y2 < ROW - 4 ; y2++) {
		for (x2 = 0 ; x2 < 64 * (COL - 4) ; x2++) {
			fprintf(fp, "%f,", D2[y2][x2]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);*/
	printf("Finish \n");
}

void searchTbl(int row, int col, char *fn) {
	char fnt[128];
    int coor[(2 * row - 1) * (2 * col - 1)][2], ix, iy, count;
    double ndis[(2 * row - 1) * (2 * col - 1)];
    sprintf(fnt, "%s_searchTbl", fn);
    
    count = 0;
    /* submit */
    for (iy = 0 ; iy < 2 * row - 1 ; iy++) {
        for (ix = 0 ; ix < 2 * col - 1 ; ix++) {
            // yCoor[iy][ix] = iy - row + 1;
            // xCoor[iy][ix] = ix - col + 1;
            coor[count][0] = iy - row + 1;
            coor[count][1] = ix - col + 1;
            ndis[count]  = sqrt(coor[count][0] * coor[count][0] + coor[count][1] * coor[count][1]);
            count++;
        }
    }
    
    int indx[(2 * row - 1) * (2 * col - 1)], tmp[(2 * row - 1) * (2 * col - 1)][2];
    for (ix = 0 ; ix < (2 * row - 1) * (2 * col - 1) ; ix++) indx[ix] = ix;
    quickSortAsc(ndis, indx, 0, (2 * row - 1) * (2 * col - 1) - 1);
    
    for (ix = 0 ; ix < (2 * row - 1) * (2 * col - 1) ; ix++) {
        tmp[ix][0] = coor[indx[ix]][0];
        tmp[ix][1] = coor[indx[ix]][1];
    }
    
    for (ix = 0 ; ix < (2 * row - 1) * (2 * col - 1) ; ix++) {
        coor[ix][0] = tmp[ix][0];
        coor[ix][1] = tmp[ix][1];
    }
    
    /* Make file pointer */
    FILE *fp;
    if((fp = fopen(fnt, "wb")) == NULL ) {
        printf("\nCannot open the file! \n");
        exit(EXIT_FAILURE);
    }
    fwrite(coor,  sizeof(int), (2 * row - 1) * (2 * col - 1) * 2, fp);
    fwrite(ndis,  sizeof(double), (2 * row - 1) * (2 * col - 1), fp);
    
    fclose(fp);
    /*
    sprintf(fnt, "%s.csv", fnt);
    if((fp = fopen(fnt, "w")) == NULL ) {
        printf("\nCannot open the file! \n");
        exit(EXIT_FAILURE);
    }
    
    for (ix = 0 ; ix < 2 ; ix++) {
        for (count = 0 ; count < (2 * row - 1) * (2 * col - 1) ; count++) {
            fprintf(fp, "%d,", coor[count][ix]);
        }
        fprintf(fp, "\n");
    }
    
    for (count = 0 ; count < (2 * row - 1) * (2 * col - 1) ; count++) {
        fprintf(fp, "%f,", ndis[count]);
    }
    
    fclose(fp); */
}

void loadTbls(double D[ROW][COL * 8], double D2[ROW - 4][(COL - 4) * 64], double ndis[(2 * ROW - 1) * (2 * COL - 1)], int coor[(2 * ROW - 1) * (2 * COL - 1)][2]) {
    /* Load table */
    char fn[128];
    FILE *fp;
    sprintf(fn, "%s/%s_DTbl", IMGDIR, RgIMAGE);
    if((fp = fopen(fn, "rb")) == NULL ) {
        printf("\nCannot open the file! \n");
        exit(EXIT_FAILURE);
    }
    fread(D, sizeof(double), ROW * COL * 8, fp);
    fclose(fp);
    printf("Finish loading DTbl.\n");
    
    sprintf(fn, "%s/%s_DTbl64", IMGDIR, RgIMAGE);
    if((fp = fopen(fn, "rb")) == NULL ) {
        printf("\nCannot open the file! \n");
        exit(EXIT_FAILURE);
    }
    fread(D2, sizeof(double), (ROW - 4) * (COL - 4) * 64, fp);
    fclose(fp);
    printf("Finish loading DTbl64.\n");

    sprintf(fn, "%s/%s_searchTbl", IMGDIR, RgIMAGE);
    if((fp = fopen(fn, "rb")) == NULL ) {
        printf("\nCannot open the file! \n");
        exit(EXIT_FAILURE);
    }
    fread(coor, sizeof(int), (2 * ROW - 1) * (2 * COL - 1) * 2, fp);
    fread(ndis, sizeof(double), (2 * ROW - 1) * (2 * COL - 1), fp);
    fclose(fp);
    printf("Finish loading searchTbl.\n");
}

void makeTemp(int g_ang2[ROW][COL], double g_can2[ROW][COL], double gt[ROW][COL], double H[ROW][COL * 162], char *fn) {
    int x1, y1, x2, y2, s, i, ix, iy, thre1, thre2;
    double tv, t0, tx2, ty2;
    double dx1, dy1, dx2, dy2;
    double gwt[ROW][COL];
    double var[] = VARTABLE;
    int count = 0;
    char fnt[128];
    sprintf(fnt, "%s_temp", fn);
    
    // double H0[ROW][COL], H1x[ROW][COL], H1y[ROW][COL];
    // double H[ROW][COL * 162];
    
    /* Loop for making template */
    for (i = 0 ; i < 6 ; i++) {
        thre1 = 27 * i * COL;
        /* update gauss */
        for (iy = 0; iy < ROW; iy++) {
            for (ix = 0; ix < COL; ix++) {
                gwt[iy][ix] = pow(gt[iy][ix], var[i]);
            }
        }
        
        for (s = -1 ; s < 8 ; s++) {
            thre2 = (s + 1) * 3 * COL;
            
            for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
                dy1 = y1 - CY;
                for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
                    dx1 = x1 - CX;
                    t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
                    for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
                        dy2 = y2 - CY;
                        for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
                            dx2 = x2 - CX;
                            
                            if (s == g_ang2[y2][x2]) {
                                tv     = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can2[y2][x2];
                                t0    += tv;
                                tx2   += tv * dx2;
                                ty2   += tv * dy2;
                            }
                            
                        }
                    }
                    
                    H[y1][thre1 + thre2 + x1]           = t0;
                    H[y1][thre1 + thre2 + COL + x1]     = tx2;
                    H[y1][thre1 + thre2 + 2 * COL + x1] = ty2;
                    
                }
            }
            
            /* write file */
            count++;
            printf("thre1 + thre2 = %d\n", thre1 + thre2);
            /*
             fwrite(H0,  sizeof(double), COL * ROW, fp);
             fwrite(H1x, sizeof(double), COL * ROW, fp);
             fwrite(H1y, sizeof(double), COL * ROW, fp);
             */
            
        }
    }
    printf("All %d times\n", count);
    /* Make file pointer */
    FILE *fp;
    if((fp = fopen(fnt, "wb")) == NULL ) {
        printf("\nCannot open the file! \n");
        exit(EXIT_FAILURE);
    }
    fwrite(H,  sizeof(double), COL * ROW * 162, fp);
    fclose(fp);
    /*
    sprintf(fnt, "%s.csv", fnt);
    if((fp = fopen(fnt, "w")) == NULL ) {
        printf("\nCannot open the file! \n");
        exit(EXIT_FAILURE);
    }
    for (y1 = 0 ; y1 < ROW ; y1++) {
        for (x1 = 0 ; x1 < 162 * COL ; x1++) {
            fprintf(fp, "%f,", H[y1][x1]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp); */
}

void makeTemp64(char sHoG2[ROW - 4][COL - 4], double g_can2[ROW][COL], double gt[ROW][COL], double H[ROW - 4][(COL - 4) * 6 * 64 * 3], char *fn) {
    int x1, y1, x2, y2, s, i, ix, iy, thre1, thre2;
	double tv, t0, tx2, ty2;
	double dx1, dy1, dx2, dy2;
	double var[6] = VARTABLE;
	double sHoGnumber[64] = sHoGNUMBER;
	double gwt[ROW][COL];
	int count = 0;
	char fnt[128];
	sprintf(fnt, "%s_temp64", fn);

	int margin = 2;
	/* Loop for making template */
	for (i = 0 ; i < 6 ; i++) {
		thre1 = 3 * 64 * i * (COL - 2 * margin);
		/* update gauss */
		for (iy = 0; iy < ROW; iy++) {
			for (ix = 0; ix < COL; ix++) {
				gwt[iy][ix] = pow(gt[iy][ix], var[i]);
			}
		}

		for (s = 0 ; s < 64 ; s++) {
			thre2 = s * 3 * (COL - 2 * margin);

			for (y1 = margin ; y1 < ROW - margin ; y1++) {
				dy1 = y1 - CY;
				for (x1 = margin ; x1 < COL - margin ; x1++) {
					dx1 = x1 - CX;
					t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
					for (y2 = margin ; y2 < ROW - margin ; y2++) {
						dy2 = y2 - CY;
						for (x2 = margin ; x2 < COL - margin ; x2++) {
							dx2 = x2 - CX;

							if (sHoGnumber[s] == sHoG2[y2 - margin][x2 - margin]) {
								tv     = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can2[y2][x2];
								t0    += tv;
								tx2   += tv * dx2;
								ty2   += tv * dy2;
							}

						}
					}

					H[y1 - margin][thre1 + thre2 + x1 - margin]                          = t0;
					H[y1 - margin][thre1 + thre2 + (COL - 2 * margin) + x1 - margin]     = tx2;
					H[y1 - margin][thre1 + thre2 + 2 * (COL - 2 * margin) + x1 - margin] = ty2;

				}
			}

			/* write file */
			count++;
			printf("thre1 + thre2 = %d\n", thre1 + thre2);

		}
	}
	printf("All %d times\n", count);
	/* Make file pointer */
	FILE *fp;
	if((fp = fopen(fnt, "wb")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	fwrite(H,  sizeof(double), (COL - 2 * margin) * (ROW - 2 * margin) * (64 * 6 * 3), fp);
	fclose(fp);
	/*
	sprintf(fnt, "%s.csv", fnt);
	if((fp = fopen(fnt, "w")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	for (y1 = 0 ; y1 < ROW - 2 * margin ; y1++) {
		for (x1 = 0 ; x1 < 1152 * (COL - 2 * margin) ; x1++) {
			fprintf(fp, "%f,", H[y1][x1]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);*/
}

void makeTemp64_far(char sHoG2[ROW - 4][COL - 4], double g_can2[ROW][COL], double gt[ROW][COL], double H[ROW - 4][(COL - 4) * 6 * 64 * 3], char *fn) {
    int x1, y1, x2, y2, s, i, ix, iy, thre1, thre2;
	double tv, t0, tx2, ty2;
	double dx1, dy1, dx2, dy2;
	double var[6] = VARTABLE2;
	double sHoGnumber[64] = sHoGNUMBER;
	double gwt[ROW][COL];
	int count = 0;
	char fnt[128];
	sprintf(fnt, "%s_temp64_far", fn);

	int margin = 2;
	/* Loop for making template */
	for (i = 0 ; i < 6 ; i++) {
		thre1 = 3 * 64 * i * (COL - 2 * margin);
		/* update gauss */
		for (iy = 0; iy < ROW; iy++) {
			for (ix = 0; ix < COL; ix++) {
				gwt[iy][ix] = pow(gt[iy][ix], var[i]);
			}
		}

		for (s = 0 ; s < 64 ; s++) {
			thre2 = s * 3 * (COL - 2 * margin);

			for (y1 = margin ; y1 < ROW - margin ; y1++) {
				dy1 = y1 - CY;
				for (x1 = margin ; x1 < COL - margin ; x1++) {
					dx1 = x1 - CX;
					t0  = 0.0; tx2 = 0.0; ty2 = 0.0;
					for (y2 = margin ; y2 < ROW - margin ; y2++) {
						dy2 = y2 - CY;
						for (x2 = margin ; x2 < COL - margin ; x2++) {
							dx2 = x2 - CX;

							if (sHoGnumber[s] == sHoG2[y2 - margin][x2 - margin]) {
								tv     = gwt[abs(y2 - y1)][abs(x2 - x1)] * g_can2[y2][x2];
								t0    += tv;
								tx2   += tv * dx2;
								ty2   += tv * dy2;
							}

						}
					}

					H[y1 - margin][thre1 + thre2 + x1 - margin]                          = t0;
					H[y1 - margin][thre1 + thre2 + (COL - 2 * margin) + x1 - margin]     = tx2;
					H[y1 - margin][thre1 + thre2 + 2 * (COL - 2 * margin) + x1 - margin] = ty2;

				}
			}

			/* write file */
			count++;
			printf("thre1 + thre2 = %d\n", thre1 + thre2);

		}
	}
	printf("All %d times\n", count);
	/* Make file pointer */
	FILE *fp;
	if((fp = fopen(fnt, "wb")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	fwrite(H,  sizeof(double), (COL - 2 * margin) * (ROW - 2 * margin) * (64 * 6 * 3), fp);
	fclose(fp);
	/*
	sprintf(fnt, "%s.csv", fnt);
	if((fp = fopen(fnt, "w")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	for (y1 = 0 ; y1 < ROW - 2 * margin ; y1++) {
		for (x1 = 0 ; x1 < 1152 * (COL - 2 * margin) ; x1++) {
			fprintf(fp, "%f,", H[y1][x1]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);*/
}

void loadTemp(double H[ROW][COL * 162]) {
	char fn[128];
	sprintf(fn, "%s/%s_temp", IMGDIR, RgIMAGE);
	FILE *fp;
	if((fp = fopen(fn, "rb")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	fread(H, sizeof(double), 162 * COL * ROW, fp);
	fclose(fp);
	printf("Finish loading temp.\n");
}

void loadTemp64(double H[ROW - 4][(COL - 4) * 6 * 64 * 3]) {
	char fn[128];
	sprintf(fn, "%s/%s_temp64", IMGDIR, RgIMAGE);
	FILE *fp;
	if((fp = fopen(fn, "rb")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	fread(H, sizeof(double), (COL - 4) * (ROW - 4) * (64 * 6 * 3), fp);
	fclose(fp);
	printf("Finish loading temp64.\n");
}

void loadTemp64_far(double H[ROW - 4][(COL - 4) * 6 * 64 * 3]) {
	char fn[128];
	sprintf(fn, "%s/%s_temp64_far", IMGDIR, RgIMAGE);
	FILE *fp;
	if((fp = fopen(fn, "rb")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	fread(H, sizeof(double), (COL - 4) * (ROW - 4) * (64 * 6 * 3), fp);
	fclose(fp);
	printf("Finish loading temp64_far.\n");
}

double fwinpat(int g_ang1[ROW][COL], int g_ang2[ROW][COL], double D[ROW][COL * 8], double ndis[(2 * ROW - 1) * (2 * COL - 1)], int coor[(2 * ROW - 1) * (2 * COL - 1)][2]) {
    /* calculation of mean of nearest-neighbor interpoint distances */
    /* with the same angle code between two images */
    double min, minInit, delta, dnn1, dnn2;
    int x1, y1, x2, y2;
    int angcode;
    int count1, count2;
    int margine = 4;
    
    minInit = sqrt((ROW - 2 * margine) * (ROW - 2 * margine) + (COL - 2 * margine) * (COL - 2 * margine));
    /* from the 1st image */
    count1 = 0;
    dnn1 = 0.0;
    for (y1 = MARGINE ; y1 < ROW - MARGINE; y1++) {
        for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
            angcode = g_ang1[y1][x1];
            if (angcode == -1) continue;
            count1++;
            min = minInit;
            /*
             for (y2 = MARGINE ; y2 < ROW - MARGINE ; y2++) {
             for (x2 = MARGINE ; x2 < COL - MARGINE ; x2++) {
             if (g_ang2[y2][x2] != angcode) continue;
             delta = (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1);
             if (delta < min) min = delta;
             }
             }
             */
            delta = D[y1][x1 + COL * angcode];
            if (delta < min) min = delta;
            // printf("angCode = %d, (%d, %d) nn1 = %f \n", angcode, x1, y1, min);
            dnn1 += min;
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
            /*
             for (y1 = MARGINE ; y1 < ROW - MARGINE ; y1++) {
             for (x1 = MARGINE ; x1 < COL - MARGINE ; x1++) {
             if (g_ang1[y1][x1] != angcode) continue;
             delta = (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1);
             if (delta < min) min = delta;
             }
             }
             */
            for (y1 = 0 ; y1 < (2 * ROW - 1) * (2 * COL - 1) ; y1++) {
                if (y2 + coor[y1][0] < 0 || y2 + coor[y1][0] >= ROW || x2 + coor[y1][1] < 0 || x2 + coor[y1][1] >= COL ) continue;
                if (g_ang1[y2 + coor[y1][0]][x2 + coor[y1][1]] != angcode) continue;
                delta = ndis[y1];
                // printf("y1 = %d nn1 = %f \n", y1, ndis[y1]);
                if (delta < min) min = delta;
                break;
            }
            dnn2 += min;
        }
    }
    dnn2 /= (double)count2;
    // printf("  count2  %d  ", count2);
    
    /* printf("Gauss parameter %f  %f  \n", dnn1, dnn2); */
    return (dnn1 + dnn2)/2.0;
}

double fsHoGpat(char sHoG1[ROW - 4][COL - 4], char sHoG2[ROW - 4][COL - 4], double D[ROW - 4][(COL - 4) * 64], double ndis[(2 * ROW - 1) * (2 * COL - 1)], int coor[(2 * ROW - 1) * (2 * COL - 1)][2]) {
    /* calculation of mean of nearest-neighbor interpoint distances */
    /* with the same angle code between two images */
    double min, minInit, delta, dnn1, dnn2;
    int x1, y1, x2, y2;
    int angcode;
    int count1, count2, sHoGnum, sHoGidx;
    int margin = 2;

    minInit = sqrt((ROW - 2 * margin) * (ROW - 2 * margin) + (COL - 2 * margin) * (COL - 2 * margin));
    /* from the 1st image */
    count1 = 0;
    dnn1 = 0.0;
    for (y1 = margin ; y1 < ROW - margin; y1++) {
        for (x1 = margin ; x1 < COL - margin ; x1++) {
        	if (sHoG1[y1 - margin][x1 - margin] == -1) continue;
            angcode = sHoG2Idx(sHoG1[y1 - margin][x1 - margin]);
            count1++;
            min = minInit;
            delta = D[y1 - margin][x1 - margin + (COL - 2 * margin) * angcode];
            if (delta < min) min = delta;
            // printf("angCode = %d, (%d, %d) nn1 = %f (%d)\n", angcode, x1, y1, min, sHoG1[y1 - margin][x1 - margin]);
            dnn1 += min;
            // printf("angCode = %d, (%d, %d) nn1 = %f (%d) \n", angcode, x1, y1, dnn1, sHoGnum);
        }
    }
    dnn1 /= (double)count1;
    // printf("  count1  %d , dnn1  %f \n", count1, dnn1);


    /* from the 2nd image */
    count2 = 0;
    dnn2 = 0.0;
    // minInit = 4.0;
    for (y2 = margin ; y2 < ROW - margin ; y2++) {
        for (x2 = margin ; x2 < COL - margin ; x2++) {
        	if (sHoG2[y2 - margin][x2 - margin] == -1) continue;
            angcode = sHoG2Idx(sHoG2[y2 - margin][x2 - margin]);
            count2++;
            min = minInit;
            for (y1 = 0 ; y1 < (2 * ROW - 1) * (2 * COL - 1) ; y1++) {
            // for (y1 = 0 ; y1 < 50 ; y1++) {
                if (y2 + coor[y1][0] < margin || y2 + coor[y1][0] >= ROW - margin || x2 + coor[y1][1] < margin || x2 + coor[y1][1] >= COL - margin ) continue;
                if (sHoG2Idx(sHoG1[y2 + coor[y1][0] - margin][x2 + coor[y1][1] - margin]) != angcode) continue;
                // if (ndis[y1] > minInit) break;
                delta = ndis[y1];
                // printf("y1 = %d nn1 = %f \n", y1, ndis[y1]);
                if (delta < min) min = delta;
                break;
            }
            dnn2 += min;
        }
    }
    dnn2 /= (double)count2;
    // printf("  count2  %d , dnn2  %f  \n", count2, dnn2);

    // printf("Gauss parameter %f  %f  \n", dnn1, dnn2);
    return (dnn1 + dnn2)/2.0;
}

void fnsgptcor(int g_ang1[ROW][COL], double g_can1[ROW][COL], double gpt[3][3], double dnn, double H[ROW][COL * 162], double Ht[ROW][COL * 27]) {
	// double H[ROW][COL * 162], Ht[ROW][COL * 27];
	double var[6] = VARTABLE;
	double newVar;

	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2, x, y, thre;
	// double gx1, gy1;
	// double g0, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3;
	// double gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2;
    double* g = (double*) malloc (G_NUM * sizeof(double));


	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double denom;
	double dx1, dx2, dy1, dy2;
	double tGpt1[3][3], tGpt2[3][3];

	int count = 0;
	newVar = 1.0 / (WGT * dnn * WGT * dnn);
	printf("newVar = %f\n", newVar);

	/* Linear interpolation */

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

			if (g_ang1[y1][x1] == -1) continue;

			thre = (g_ang1[y1][x1] + 1) * 3 * COL;

			t0     = Ht[y1][thre + x1]           * g_can1[y1][x1];
			tx2    = Ht[y1][thre + x1 + COL]     * g_can1[y1][x1];
			ty2    = Ht[y1][thre + x1 + COL * 2] * g_can1[y1][x1];

            // printf("(%d %d) t0 = %f \n", y1, x1, t0);

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

    // printf("g0 = %f\n", g0);

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
}

void fnsgptcorSpHOG5x5(int g_ang1[ROW][COL], char sHoG1[ROW - 4][COL - 4], double g_can1[ROW][COL],
                     double gpt[3][3], double dnn, double H[ROW - 4][(COL - 4) * 6 * 64 * 3], double Ht[ROW - 4][(COL - 4) * 64 * 3]) {
	// double H[ROW - 4][(COL - 4) * 6 * 64 * 3], Ht[ROW - 4][(COL - 4) * 64 * 3];
	double var[6] = VARTABLE;
	double newVar;
	double sHoGnumber[64] = sHoGNUMBER;
	int margin = 2;

	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2, x, y, thre, s;
	// double gx1, gy1;
	// double g0, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3;
	// double gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2;
    double* g = (double*) malloc (G_NUM * sizeof(double));

	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double denom;
	double dx1, dx2, dy1, dy2;
	double tGpt1[3][3], tGpt2[3][3];
    

	int count = 0;
	newVar = 1.0 / (WGT * dnn * WGT * dnn);

	/* Linear interpolation */

    if(isGPU){
        cuda_update_parameter(g_ang1, g_can1, H,sHoG1);
        cuda_Ht(newVar);
        g = cuda_calc_g();
        
    } else {
        if (newVar > 1.0) {
            for (y = 0 ; y < ROW - 2 * margin ; y++) {
                for (x = 0 ; x < 3 * 64 * (COL - 2 * margin) ; x++) {
                        Ht[y][x] =  H[y][x + (COL - 2 * margin) * 3 * 64 * 5];
                }
            }
        } else if (newVar < 1.0 / 32.0) {
            for (y = 0 ; y < ROW - 2 * margin ; y++) {
                for (x = 0 ; x < 3 * 64 * (COL - 2 * margin) ; x++) {
                        Ht[y][x] =  H[y][x];
                }
            }
        } else {
            count = floor(log2(newVar)) + 5;
            for (y = 0 ; y < ROW - 2 * margin ; y++) {
                for (x = 0 ; x < 3 * 64 * (COL - 2 * margin) ; x++) {
                        Ht[y][x] =  H[y][x + (COL - 2 * margin) * 3 * 64 * count] +
                                   (H[y][x + (COL - 2 * margin) * 3 * 64 * (count + 1)] - H[y][x + (COL - 2 * margin) * 3 * 64 * count])
                                 / (var[count + 1] - var[count])
                                 * (newVar - var[count]);
                        // printf("Ht = %f H[count] = %f H[count + 1] = %f \n", Ht[y][x], H[y][x + COL * 27 * count], H[y][x + COL * 27 * (count + 1)]);
                }
            }
        }
    

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

                thre = -1;
                for (s = 0 ; s < 64 ; s++) {
                    if (sHoG1[y1 - margin][x1 - margin] == sHoGnumber[s]) {
                        thre = s * 3 * (COL - 2 * margin);
                        break;
                    }
                }
                if (thre == -1) {
                    printf("ERROR! \n");
                }

                t0     = Ht[y1 - margin][thre + x1 - margin]                          * g_can1[y1][x1];
                tx2    = Ht[y1 - margin][thre + x1 - margin + (COL - 2 * margin)]     * g_can1[y1][x1];
                ty2    = Ht[y1 - margin][thre + x1 - margin + (COL - 2 * margin) * 2] * g_can1[y1][x1];

                // printf("(%d %d) t0 = %f \n", y1, x1, t0);

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
    }

	
    for(int i=0;i<G_NUM;i++){
        cout<<g[i]<<" ";
    }
    cout<<endl;
	

    // printf("g0 = %f\n", g0);

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

   
}

void fnsgptcorSpHOG5x5_far(int g_ang1[ROW][COL], char sHoG1[ROW - 4][COL - 4], double g_can1[ROW][COL],
                     double gpt[3][3], double dnn, double H[ROW - 4][(COL - 4) * 6 * 64 * 3], double Ht[ROW - 4][(COL - 4) * 64 * 3]) {
	// double H[ROW - 4][(COL - 4) * 6 * 64 * 3], Ht[ROW - 4][(COL - 4) * 64 * 3];
	double var[6] = VARTABLE2;
	double newVar;
	double sHoGnumber[64] = sHoGNUMBER;
	int margin = 2;

	/* determination of optimal GAT components */
	/* that yield the maximal correlation value */
	int x1, y1, x2, y2, x, y, thre, s;
	// double gx1, gy1;
	// double g0, gx1p1, gx1p2, gx1p3, gx1p4, gy1p1, gy1p2, gy1p3, gy1p4, gx1p1y1p1, gx1p2y1p1, gx1p3y1p1, gx1p1y1p2, gx1p2y1p2, gx1p1y1p3;
	// double gx1x2, gy1x2, gx1y2, gy1y2, gx2, gy2, gx1p2x2, gx1y1y2, gx1y1x2, gy1p2y2;
    double* g = (double*) malloc (G_NUM * sizeof(double));

	double tv, t0, tx2, ty2, gx2x2, gx2y2, gy2y2;
	double denom;
	double dx1, dx2, dy1, dy2;
	double tGpt1[3][3], tGpt2[3][3];

	int count = 0;
	newVar = 1.0 / (WGT * dnn * WGT * dnn);

	/* Linear interpolation */

	if (newVar > var[5]) {
		for (y = 0 ; y < ROW - 2 * margin ; y++) {
			for (x = 0 ; x < 3 * 64 * (COL - 2 * margin) ; x++) {
					Ht[y][x] =  H[y][x + (COL - 2 * margin) * 3 * 64 * 5];
			}
		}
	} else if (newVar < var[0]) {
		for (y = 0 ; y < ROW - 2 * margin ; y++) {
			for (x = 0 ; x < 3 * 64 * (COL - 2 * margin) ; x++) {
					Ht[y][x] =  H[y][x];
			}
		}
	} else {
		count = floor(log2(newVar)) + 10;
		printf("count = %d\n", count);
		for (y = 0 ; y < ROW - 2 * margin ; y++) {
			for (x = 0 ; x < 3 * 64 * (COL - 2 * margin) ; x++) {
					Ht[y][x] =  H[y][x + (COL - 2 * margin) * 3 * 64 * count] +
							   (H[y][x + (COL - 2 * margin) * 3 * 64 * (count + 1)] - H[y][x + (COL - 2 * margin) * 3 * 64 * count])
							 / (var[count + 1] - var[count])
							 * (newVar - var[count]);
					// printf("Ht = %f H[count] = %f H[count + 1] = %f \n", Ht[y][x], H[y][x + COL * 27 * count], H[y][x + COL * 27 * (count + 1)]);
			}
		}
	}

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

			thre = -1;
			for (s = 0 ; s < 64 ; s++) {
				if (sHoG1[y1 - margin][x1 - margin] == sHoGnumber[s]) {
					thre = s * 3 * (COL - 2 * margin);
					break;
				}
			}
			if (thre == -1) {
				printf("ERROR! \n");
			}

			t0     = Ht[y1 - margin][thre + x1 - margin]                          * g_can1[y1][x1];
			tx2    = Ht[y1 - margin][thre + x1 - margin + (COL - 2 * margin)]     * g_can1[y1][x1];
			ty2    = Ht[y1 - margin][thre + x1 - margin + (COL - 2 * margin) * 2] * g_can1[y1][x1];

            // printf("(%d %d) t0 = %f \n", y1, x1, t0);

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

    // printf("g0 = %f\n", g0);

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
}
