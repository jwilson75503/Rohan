/* standard libraries */
#include <conio.h> //for _getch 
#include <iostream>
#include <float.h>
#include <math.h>  //for M_PI
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>
#include <sys/timeb.h>

/* CUDA-relevant includes */
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <multithreading.h>

/* Project includes */
#include "Rohan.h"

extern int iDebugLvl, iDevDebug, iTrace;
extern float gElapsedTime, gKernelTimeTally;

// device-global variables to facilitate data transfer
//__device__ __align__(16) __constant__ struct rohanContext devSes;
//__device__ __align__(16) __constant__ struct rohanLearningSet devLearn;
//__device__ __align__(16) struct rohanNetwork devNet;
//__device__ __align__(16) const cuDoubleComplex gpuZero = { 0, 0 };
//__device__ __align__(16) double devdReturn[1024*1024];
//__device__ __align__(16) double devdRMSE=0;
//__device__ __align__(16) int devlReturn[1024*1024];
//__device__ __align__(16) int devlTrainable=0;
//__device__ __align__(16) int iDevDebug=0;
