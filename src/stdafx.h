// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently

#pragma once

#include "targetver.h"

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

// Shared Utilities (QA Testing)
#include <shrUtils.h>
#include <shrQATest.h>

/* Project includes */
#include "cBarge.h"
#include "cDrover.h"
#include "Complex-math-func.h"
#include "cTeam.h"
#include "Rohan.h"
#include "Rohan-data.h"
#include "Rohan-io.h"
#include "Rohan-kernel.h"
#include "Rohan-learn.h"
#include "Rohan-menu.h"
#include "ShowMe.h"


// defines
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))