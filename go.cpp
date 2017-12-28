#include<iostream>
#include <stdlib.h> 
#include <ctime>
#include<stdio.h>

using namespace std;

#define NX_F 170 * 2
#define NY_F 136 * 2

#define NX_G 170 * 2
#define NY_G 136*8 * 2



int main(){
	cout<<"NY_G: "<<NY_G<<endl;
	srand(1);
	int f[NY_F][NX_F];
	int g[NY_G][NX_G];
	int h =0 ;
	for(int i=0;i<NY_F;i++){
		for(int j=0;j<NX_F;j++){
			f[i][j]=rand()%9-1;
		}
	}
	for(int i=0;i<NY_G;i++){
		for(int j=0;j<NX_G;j++){
			g[i][j]=rand()%19-1;
		}
	}

	clock_t begin = clock();

	for(int i=0;i<NY_F;i++){
		for(int j=0;j<NX_F;j++){
			if(f[i][j]<0)continue;
			h+=g[i][f[i][j]*10+j];
		}
	}

	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;
  	printf("Time elapsed: %.7f ms\n",elapsed_secs);
  	printf("h: %d\n",h);

	return 0;
}