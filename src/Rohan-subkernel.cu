/* Includes, cuda */
#include <cuda.h>
#include "cublas.h"
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include "cuPrintf.cuh"
#include "cuPrintf1.cuh"
#include "Rohan.h"
#include "Rohan-kernel.h"

#include "crc.h"
#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/timeb.h>

//#define START_SIZE 10
//#define STOP_SIZE 40000 
//#define BLOCK_SIZE 256 
#define ONE_OVER_TWO_PI 0.15915494309189533576888376337254

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define mThread(t) ((t-1)/32+1),(t-1)%32+1

extern int iDebugLvl, iWarnings, iErrors, iTrace;

__device__ double devdReturn;
/// device-global variables to facilitate data transfer
__device__ struct rohanContext devSes;
__device__ struct rohanLearningSet devLearn;
__device__ struct rohanNetwork devNet;
__device__ void * devPointer;
__device__ long devLong0;
__device__ long devLong1;
__device__ long devLong2;
__device__ long devLong3;
__device__ long devLong4;
__device__ long devLong5;
__device__ long devLong6;
__device__ long devLong7;
__device__ long devLong8;
__device__ long devLong9;
__device__ double devDouble0;
__device__ double devDouble1;
__device__ cuDoubleComplex devCDC;
__device__ char devString[256];



