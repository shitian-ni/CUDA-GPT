#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include<iostream>


#include"include/parameter.h"
#include"include/utility.h"
#include"include/stdGpt.h"
#include"include/acclGpt.h"
#include "include/acclGpt_cuda.h"

using namespace std;

/* Image storage arrays */
unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE];
unsigned char image2[MAX_IMAGESIZE][MAX_IMAGESIZE];
int x_size1 = COL, y_size1 = ROW; /* width & height of image1*/
int x_size2, y_size2; /* width & height of image2 */

double H1[ROW][COL * 162], Ht1[ROW][COL * 27];
double H2[ROW - 4][(COL - 4) * 6 * 64 * 3], Ht2[ROW - 4][(COL - 4) * 64 * 3];
double H3[ROW - 4][(COL - 4) * 6 * 64 * 3], Ht3[ROW - 4][(COL - 4) * 64 * 3];
double D1[ROW][COL * 8];
double D2[ROW - 4][(COL - 4) * 64];

double ndis[(2 * ROW - 1) * (2 * COL - 1)];
int coor[(2 * ROW - 1) * (2 * COL - 1)][2];

int main() {
	int image3[ROW2][COL2], image4[ROW][COL];					// image3: test image	image4: training image
	int x, y, iter;
	char csvname[MAX_FILENAME], foldername[MAX_FILENAME];	// GAT, NGAT, GPT, NGPT, name of .csv file, foldername
	double gk[ROW][COL], gwt[ROW][COL], dnn, temp_dnn, var;			// Gaussian window initial, Gaussian window, window size, variance
	int g_ang1[ROW][COL], g_ang2[ROW][COL];					// direction of gradients
	char g_HoG1[ROW][COL][8], g_HoG2[ROW][COL][8];			// HoG feature of the images
	char sHoG1[ROW - 4][COL - 4], sHoG2[ROW - 4][COL - 4];
	double g_nor1[ROW][COL], g_nor2[ROW][COL];				// norm of gradients
	double g_can1[ROW][COL], g_can2[ROW][COL];				// canonicalized images
	double g_can11[ROW - CANMARGIN][COL - CANMARGIN], g_can22[ROW - CANMARGIN][COL - CANMARGIN];
															// canonicalized images center
	double old_cor0, old_cor1, new_cor1;					//
	double org_cor, gat_corf, gat_corb;
	double gpt0[3][3], gpt1[3][3], gptInv[3][3];

	clock_t start, end;
	double elapse;

	char fileName[128];


	cuda_init_parameter();

	/* Initialize the GPT matrix */
	initGpt(gpt0);
	initGpt2(gpt1, ZOOM, ZOOM*BETA, B1, B2, ROT);

	/* initialize Gauss window function */
	for (y = 0; y < ROW; y++)
		for (x = 0; x < COL; x++)
			gk[y][x] = exp(-(x*x+y*y)/2.0);

	/* Load template image and save it to image4, the local memory */
	sprintf(fileName, "%s/%s.pgm", IMGDIR, RgIMAGE);
	load_image_file(fileName, image1, COL, ROW);
	for (y = 0; y < ROW; y++)
		for (x = 0; x < COL; x++)
			image4[y][x] = image1[y][x];
	procImg(g_can2, g_ang2, g_nor2, g_HoG2, sHoG2, image1);
    
    /* Make template tables if required */
#if MAKETEMP != 0
	sprintf(fileName, "%s/%s", IMGDIR, RgIMAGE);
	makeTemp(g_ang2, g_can2, gk, H1, fileName);
	makeTemp64(sHoG2, g_can2, gk, H2, fileName);
	// makeTemp64_far(sHoG2, g_can2, gk, H3, fileName);
    winTbl(g_ang2, D1, fileName);
    winTbl64(sHoG2, D2, fileName);
    searchTbl(ROW, COL, fileName);
    return 0;
#else
    loadTbls(D1, D2, ndis, coor);
    loadTemp(H1);
    loadTemp64(H2);
    loadTemp64_far(H3);
#endif

    init_gk_and_g_can2_and_H2(gk,g_can2,H2);

	/* Load test image and save it to image3, the local memory */
	sprintf(fileName, "%s/%s.pgm", IMGDIR, TsIMAGE);
	load_image_file(fileName, image1, COL2, ROW2);


	for (y = 0; y < ROW2; y++)
		for (x = 0; x < COL2; x++)
			image3[y][x] = image1[y][x];

	/* save the initial image */
	for (y = 0; y < ROW2; y++)
		for (x = 0; x < COL2; x++)
			image2[y][x] = image1[y][x];
	bilinear_normal_projection(gpt1, COL, ROW, COL2, ROW2, image1, image2);
	sprintf(fileName, "%s/%s_init.pgm", IMGDIR, RgIMAGE);
	save_image_file(fileName, image2, COL, ROW);
	procImg(g_can1, g_ang1, g_nor1, g_HoG1, sHoG1, image2);

	/***************Pre-setting finish***************/
    
	/* calculate the initial correlation */
	old_cor1 = 0.0;
	for (y = MARGINE ; y < ROW - MARGINE ; y++)
		for (x = MARGINE ; x < COL - MARGINE ; x++)
			old_cor1 += g_can1[y][x] * g_can2[y][x];
	org_cor = old_cor1;
	printf("Original cor. = %f\n", org_cor);
	old_cor0 = old_cor1;

	/* calculate the initial dnn */
	switch (DISTANCETYPE) {
	case 0:
		dnn = winpat(g_ang1, g_ang2);
		if (dnn > DNNSWITCHTHRE)
			dnn = sHoGpat(sHoG1, sHoG2);
		break;
	case 1:
		dnn = winpat(g_ang1, g_ang2);
		break;
	case 2:
		dnn = fwinpat(g_ang1, g_ang2, D1, ndis, coor);
		break;
	case 3:
		dnn = sHoGpat(sHoG1, sHoG2);
		break;
	case 4:
		dnn = fsHoGpat(sHoG1, sHoG2, D2, ndis, coor);
		break;
	case 10:
		dnn = fwinpat(g_ang1, g_ang2, D1, ndis, coor);
		if (dnn > DNNSWITCHTHRE)
			dnn = fsHoGpat(sHoG1, sHoG2, D2, ndis, coor);
		break;
	}


	/***************Main iteration loop*************/
	/* lap the start time */
	start = clock();
	for (iter = 0 ; iter < MAXITER ; iter++) {
		/* Calculation distance */
		switch (DISTANCETYPE) {
		case 0:
			if (dnn < DNNSWITCHTHRE) {
				dnn = winpat(g_ang1, g_ang2);
			} else {
				dnn = sHoGpat(sHoG1, sHoG2);
			}
			break;
		case 1:
			dnn = winpat(g_ang1, g_ang2);
			break;
		case 2:
			dnn = fwinpat(g_ang1, g_ang2, D1, ndis, coor);
			break;
		case 3:
			dnn = sHoGpat(sHoG1, sHoG2);
			break;
		case 4:
			dnn = fsHoGpat(sHoG1, sHoG2, D2, ndis, coor);
			break;
		case 10:
			dnn = fwinpat(g_ang1, g_ang2, D1, ndis, coor);
			if (dnn > DNNSWITCHTHRE)
				dnn = fsHoGpat(sHoG1, sHoG2, D2, ndis, coor);
			break;
		}

		/* update gauss window function */
		var = pow(WGT * dnn, 2);
		#if isGPU == 0
			for (y = 0; y < ROW; y++)
				for (x = 0; x < COL; x++)
					gwt[y][x] = pow(gk[y][x], 1.0 / var);
		#elif isGPU == 1
			calc_gwt(var, gwt);
		#endif

		/* select matching method */
		switch (MATCHMETHOD) {
		case 1:
			break;
		case 6:
			nsgptcor(g_ang1, g_can1, g_ang2, g_can2, gwt, gpt1, dnn);
			break;
		case 7:
			nsgptcorSpHOG5x5(g_ang1, sHoG1, g_can1,	g_ang2, sHoG2, g_can2, gwt, gpt1, dnn);
			break;
		case 16:
			fnsgptcor(g_ang1, g_can1, gpt1, dnn, H1, Ht1);
			break;
		case 17:
			fnsgptcorSpHOG5x5(g_ang1, sHoG1, g_can1, gpt1, dnn, H2, Ht2);
			// fnsgptcorSpHOG5x5_far(g_ang1, sHoG1, g_can1, gpt1, dnn, H3, Ht3);
		}


		/* transform the test image and update g_can1, g_ang1, g_nor1, g_HoG1, sHoG1 */
		for (y = 0; y < ROW2; y++)
			for (x = 0; x < COL2; x++)
				image1[y][x] = (unsigned char)image3[y][x];
		bilinear_normal_projection(gpt1, COL, ROW, COL2, ROW2, image1, image2);

		procImg(g_can1, g_ang1, g_nor1, g_HoG1, sHoG1, image2);

		/* update correlation */
		#if isGPU == 0
			new_cor1 = 0.0;
			for (y = MARGINE ; y < ROW - MARGINE ; y++){
				for (x = MARGINE ; x < COL - MARGINE ; x++){
					new_cor1 += g_can1[y][x] * g_can2[y][x];
				}
			}
		#elif isGPU == 1
			new_cor1 = calc_new_cor1();
		#endif
		

		/* display message */
		printf("iter = %d, new col. = %f dnn = %f  var = %f\n", iter, new_cor1, dnn, 1 / var);

	}
	/* display the calculation time */
	end = clock();
	elapse = (double)(end - start) / CLOCKS_PER_SEC;
	string device = isGPU?"GPU":"CPU";
	printf("\n%s elapsed time = %.3f sec\n\n", device.c_str(),elapse);
}
