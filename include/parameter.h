#ifndef PARAMETER_H
#define PARAMETER_H

// #include"accGpt.h"

/* switch match method 
 1: GAT matching, the conventional GAT matching
 2: GPT matching, the conventional GPT matching by alternative calculation
 3: NGAT matching, the GAT matching with norm normalization
 4: NGPT matching, the GPT matching with norm normalization
 5: SGPT matching, the conventional enhanced GPT matching
 6: NSGPT matching, the enhanced GPT matching with norm normalization
 7: NSGPT-sHOG matching, the enhanced GPT matching with norm normalization 
    associated with simplified HOG patterns
 11: FGAT (F + matching method = acceleration algorithm)
 12: FGPT
 13: FNGAT
 14: FNGPT
 15: FSGPT
 16: FNSGPT
 17: FNSGPT-sHOG
 21: GPU-FGAT (acceleration algorithm via GPU parallel calculation)
 22: GPU-FGPT (acceleration algorithm via GPU parallel calculation)
 23: GPU-FNGAT (acceleration algorithm via GPU parallel calculation)
 24: GPU-FNGPT (acceleration algorithm via GPU parallel calculation)
 25: GPU-FSGPT (acceleration algorithm via GPU parallel calculation)
 26: GPU-FNSGPT (acceleration algorithm via GPU parallel calculation)
 27: GPU-FNSGPT-sHOG (acceleration algorithm via GPU parallel calculation)
 */
#define MATCHMETHOD 17

/* switch calculation type 
 0: use CPU to calculate defcan, roberts8, calHoG, and smpHoG64
 1: use GPU to calculate defcan, roberts8, calHoG, and smpHoG64
 */
#define isGPU 1

/* CUDA parameters */
#define TPB 32
#define TPB_X_TPB TPB*TPB
#define G_NUM 30

#define g0 g[0]
#define gx1 g[1]
#define gy1 g[2]
#define gx1p1 g[3]
#define gx1p2 g[4]
#define gx1p3 g[5]
#define gx1p4 g[6]
#define gy1p1 g[7]
#define gy1p2 g[8]
#define gy1p3 g[9]
#define gy1p4 g[10]
#define gx1p1y1p1 g[11]
#define gx1p2y1p1 g[12]
#define gx1p3y1p1 g[13]
#define gx1p1y1p2 g[14]
#define gx1p2y1p2 g[15]
#define gx1p1y1p3 g[16]
#define gx1x2 g[17]
#define gy1x2 g[18]
#define gx1y2 g[19]
#define gy1y2 g[20]
#define gx2 g[21]
#define gy2 g[22]
#define gx1p2x2 g[23]
#define gx1y1y2 g[24]
#define gx1y1x2 g[25]
#define gy1p2y2 g[26]
#define gx1x1 g[27]
#define gx1y1 g[28]
#define gy1y1 g[29]


#define ROW_H1 (ROW)
#define COL_H1 (COL * 162)
#define COL_Ht1 (COL * 27)

#define ROW_H2 (ROW-4)
#define COL_H2 ((COL - 4) * 6 * 64 * 3)
#define COL_Ht2 ((COL - 4) * 64 * 3)

#define ROW_H3 (ROW-4)
#define COL_H3 ((COL - 4) * 6 * 64 * 3)
#define COL_Ht3 ((COL - 4) * 64 * 3)

/* switch template table type 
 0: automatical selection of the type of window size
 1: for 8-quantized gradient directions
 2: for acceleration calculation of 8-quantized gradient directions
 3: for simplified HOG patterns
 4: for acceleration calculation simplified HOG patterns
 10: automatical selection of the type of window size using acceleration algorithm
 */
#define DISTANCETYPE 10

/* Execute parameters */
#define MAXITER 30			// Maximum iteration times
#define MAXNR 5					// Maximum Newton-Raphson iterations

/* --------------Fixed parameters-------------- */
#define BLACK 0					// color of black
#define WHITE 255				// color of white
#define NoDIRECTION		20.0	// The threshold of non gradient direction if the norm of gradient is weak
#define NOHoG			8		// The threshold of non HoG feature
#define SHoGTHRE        300.0   // The first direction should over this value
#define SHoGSECONDTHRE	0.5		// The threshold of the second direction of the simplified HoG pattern
#define DNNSWITCHTHRE	3.0		// The threshold of switch the method of dnn calculation
#define TRUNC			(2 * COL - 1) * (2 * ROW - 1)
#define PI 3.141592654			// value of pi
#define EPS 0.000001         	// like zero
#define EPS2 0.000001           // like zero2
#define WGT 1.5       			/* Gauss window size weight */
#define MAX_IMAGESIZE  1024		//
#define MAX_BRIGHTNESS  255 	/* Maximum gray level */
#define GRAYLEVEL       256 	/* No. of gray levels */
#define MAX_FILENAME    256 	/* Filename length limit */
#define MAX_BUFFERSIZE  256
#define HoGTHRESHOLD	8
#define HoGTHRESHOLD3	3
#define TRUE			1
#define FALSE			0
/* --------------Fixed parameters-------------- */

/* initial conditions */
#define NONELEMENT				/* use non-elemental matrix as initial condition */
#define ZOOM			0.7		/* Zoom rate for initial matrix */
#define BETA			2.0		/* Relation between alpha and beta */
#define ROT				-45.0		/* Rotation angle for initial matrix */
#define B1				0.0		/*  */
#define B2				20.0		/*  */

/* information of window sizes and ID of simplified HOG patterns */
#define VARTABLE  {1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0};
#define VARTABLE2 {1.0 / 1024, 1.0 / 512, 1.0 / 256, 1.0 / 128, 1.0 / 64, 1.0 / 32}
#define sHoGNUMBER {1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 21, 23, 24, 25, 26, 27, 28, 31, 32, 34, 35, 36, 37, 38, 41, 42, 43, 45, 46, 47, 48, 51, 52, 53, 54, 56, 57, 58, 61, 62, 63, 64, 65, 67, 68, 71, 72, 73, 74, 75, 76, 78, 81, 82, 83, 84, 85, 86, 87};

/* if defined, make reference tables
 0 := do not make table
 1 := make all tables
 */
#define MAKETEMP 0


/********************* Image files' information *********************/
// #define IMGDIR          "/home2/zhang/gatWork/sGpt/zhangGpt/ztmp"
#define IMGDIR          "."
// #define SAVEDATAFORMATLAB	// if defined, save .csv data for matlab plot

/* Dataset type 
 0: MNIST
 1: CHESS       // 20, 400, 0.85
 2: BOAT
 3: GRAF        // 20, 400, 0.5
 4: WALL
 */
#define DATATYPE 3

#if DATATYPE == 1
#define COL 28          		/* Horizontal size of image  */
#define ROW 28          		/* Vertical size of image    */
#define CX  14					/* Center of x               */
#define CY  14					/* Center of y               */

#define MARGINE 0				/* Margine size              */
#define TsIMAGE  "tdg0_0001_gray"
#define RgIMAGE  "ldg0_0001_gray"
#endif

#if DATATYPE == 2
#define COL 170           /* Horizontal size of image  */
#define ROW 136           /* Vertical   size of image  */
#define COL2 340          /* Horizontal size of image  */
#define ROW2 272          /* Vertical   size of image  */
#define CX  85
#define CY  68
#define CX2  170
#define CY2  136

#define MARGINE 0				/* Margine size              */
#define CANMARGIN 0			/* Margine size for calculate crr */
#define TsIMAGE  "sample_boat/img3_small2"
#define RgIMAGE  "sample_boat/img1_small"
#define CENTERCORRELATION
/* acc: 3.5,
 * Memo of the parameter set
 * img1_small to img2_small2 --> ZOOM = 2.0;
 * img1_small to img3_small2 --> ZOOM = 2.0, ROT = 45.0;
 * img1_small to img4_small2 --> ZOOM = 1.4, ROT = 90.0;
 * img1_small to img5_small2 --> ZOOM = 1.1;
 * img1_small to img6_small2 --> ZOOM = 0.9, ROT = 45.0;
 */
#endif


#if DATATYPE == 3
#define COL 160          /* Horizontal size of image  */
#define ROW 128          /* Vertical size of image    */
#define COL2 320          /* Horizontal size of image  */
#define ROW2 256          /* Vertical size of image    */
#define CX  80
#define CY  64
#define CX2 160
#define CY2 128

#define MARGINE 0				/* Margine size              */
#define CANMARGIN 0			/* Margine size for calculate crr */
#define TsIMAGE  "sample_graf/img6_small2"
#define RgIMAGE  "sample_graf/img1_small"
#define CENTERCORRELATION
/*
 * Memo of the parameter set
 * img1_small to img2_small2 --> ZOOM = 1.6;
 * img1_small to img3_small2 --> ZOOM = 1.6;
 * img1_small to img4_small2 --> ZOOM = 1.6;
 * img1_small to img5_small2 --> ZOOM = 1.0;
 * img1_small to img6_small2 --> ZOOM = 0.9, ROT = -45.0, BETA = 2, B2 = 20;
 */

#endif

#if DATATYPE == 4
#define COL 200          /* Horizontal size of image  */
#define ROW 140          /* Vertical size of image    */
#define COL2 352          /* Horizontal size of image  */
#define ROW2 272          /* Vertical size of image    */
#define CX  100
#define CY  70
#define CX2 176
#define CY2 136

#define MARGINE 0				/* Margine size              */
#define CANMARGIN 0			/* Margine size for calculate crr */
#define TsIMAGE  "sample_wall/img2_small2"
#define RgIMAGE  "sample_wall/img1_small"
#define CENTERCORRELATION

#endif

#if DATATYPE == 5
#define COL 160          /* Horizontal size of image  */
#define ROW 128          /* Vertical size of image    */
#define COL2 160          /* Horizontal size of image  */
#define ROW2 128          /* Vertical size of image    */
#define CX  80
#define CY  64
#define CX2 80
#define CY2 64

#define MARGINE 0				/* Margine size              */
#define CANMARGIN 0			/* Margine size for calculate crr */
#define TsIMAGE  "sample_ubc/img6_small"
#define RgIMAGE  "sample_ubc/img1_small"
#define CENTERCORRELATION

#endif

#if DATATYPE == 6
#define COL 360          /* Horizontal size of image  */
#define ROW 240          /* Vertical size of image    */
#define COL2 360          /* Horizontal size of image  */
#define ROW2 248          /* Vertical size of image    */
#define CX  180
#define CY  120
#define CX2 180
#define CY2 120

#define MARGINE 0				/* Margine size              */
#define CANMARGIN 0			/* Margine size for calculate crr */
#define TsIMAGE  "sample_leuven/img6_small"
#define RgIMAGE  "sample_leuven/img1_small"
#define CENTERCORRELATION

#endif

#if DATATYPE == 7		/* 10, 100 */
#define COL 200          /* Horizontal size of image  */
#define ROW 140          /* Vertical size of image    */
#define COL2 200          /* Horizontal size of image  */
#define ROW2 140          /* Vertical size of image    */
#define CX  100
#define CY  70
#define CX2 100
#define CY2 70

#define MARGINE 0				/* Margine size              */
#define CANMARGIN 0			/* Margine size for calculate crr */
#define TsIMAGE  "sample_bikes/img6_small"
#define RgIMAGE  "sample_bikes/img1_small"
#define CENTERCORRELATION

#endif

#if DATATYPE == 8		/* 10, 100 */
#define COL 200          /* Horizontal size of image  */
#define ROW 140          /* Vertical size of image    */
#define COL2 200          /* Horizontal size of image  */
#define ROW2 140          /* Vertical size of image    */
#define CX  100
#define CY  70
#define CX2 100
#define CY2 70

#define MARGINE 0				/* Margine size              */
#define CANMARGIN 0			/* Margine size for calculate crr */
#define TsIMAGE  "sample_tree/img6_small"
#define RgIMAGE  "sample_tree/img1_small"
#define CENTERCORRELATION

#endif

#define ROW_X_COL ROW*COL

#endif // PARAMETER_H
