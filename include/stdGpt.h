#ifndef STDGPT_H
#define STDGPT_H

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
#endif
