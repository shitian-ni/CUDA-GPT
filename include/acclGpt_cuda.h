void cuda_init_parameter();
void cuda_update_parameter(char sHoG1[ROW - 4][COL - 4]);
void cuda_update_image(unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]);
void cuda_Ht(double newVar,int H_No);
double* cuda_calc_g(int calc_g_type);
void cuda_calc_defcan1(double g_can1[ROW][COL],unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]);
void cuda_procImg(double g_can[ROW][COL], int g_ang[ROW][COL], double g_nor[ROW][COL], char g_HoG[ROW][COL][8], char sHoG[ROW - 4][COL - 4], unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]);
void copy_initial_parameters(double gk[ROW][COL],double g_can2[ROW][COL],double H1[ROW_H1][COL_H1],double H2[ROW_H2][COL_H2],double H3[ROW_H3][COL_H3],
	double D1[ROW][COL * 8], double D2[ROW - 4][(COL - 4) * 64],char sHoG2[ROW - 4][COL - 4],double ndis[(2 * ROW - 1) * (2 * COL - 1)],
int coor[(2 * ROW - 1) * (2 * COL - 1)][2]);
double cuda_fsHoGpat(char sHoG1[ROW - 4][COL - 4]);

void calc_gwt(double var,double gwt[ROW][COL]);
double calc_new_cor1();
void cuda_bilinear_normal_inverse_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
		unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE], unsigned char image2[MAX_IMAGESIZE][MAX_IMAGESIZE]);