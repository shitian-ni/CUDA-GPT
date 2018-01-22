void cuda_init_parameter(unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]);
void cuda_update_parameter(int g_ang1[ROW][COL], double g_can1[ROW][COL],double H[ROW_H][COL_H],char sHoG1[ROW - 4][COL - 4]);

void cuda_Ht(double newVar);
double* cuda_calc_g();