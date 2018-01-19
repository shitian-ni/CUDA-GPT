#ifndef UTILITY_H
#define UTILITY_H

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

#endif
