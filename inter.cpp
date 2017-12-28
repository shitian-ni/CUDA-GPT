#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<iostream>

using namespace std;

#define ROW 64
#define COL 64

double varTable[6] = {1.0 / 32, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0};
double H[ROW][COL * 162], Ht[ROW][COL * 27];

int main(){
	double newVar = 1.0/3;
	char fileName[128];
	sprintf(fileName, "tests0_temp");
	FILE *fp;
	if((fp = fopen(fileName, "rb")) == NULL ) {
		printf("\nCannot open the file! \n");
		exit(EXIT_FAILURE);
	}
	fread(H, sizeof(double), 162 * COL * ROW, fp);


	
	if (newVar > 1.0) {
		for (int y = 0 ; y < ROW ; y++) {
			for (int x = 0 ; x < 27 * COL ; x++) {
					Ht[y][x] =  H[y][x + COL * 27 * 5];
			}
		}
	} else if (newVar < 1.0 / 32.0) {
		for (int y = 0 ; y < ROW ; y++) {
			for (int x = 0 ; x < 27 * COL ; x++) {
					Ht[y][x] =  H[y][x];
			}
		}
	} else {
		int count = floor(log2(newVar)) + 5;
		for (int y = 0 ; y < ROW ; y++) {
			for (int x = 0 ; x < 27 * COL ; x++) {
					Ht[y][x] =  H[y][x + COL * 27 * count] + (H[y][x + COL * 27 * (count + 1)] - H[y][x + COL * 27 * count]) / (varTable[count + 1] - varTable[count]) * (newVar - varTable[count]);
					// printf("Ht = %f H[count] = %f H[count + 1] = %f \n", Ht[y][x], H[y][x + COL * 27 * count], H[y][x + COL * 27 * (count + 1)]);
			}
		}
	}
	freopen("cpp_Ht.txt","w",stdout);
	for(int i=0;i<ROW;i++){
		for(int j=0;j<27*COL;j++){
			cout<<Ht[i][j]<<" ";
		}
		cout<<endl;
	}
	return 0;
}