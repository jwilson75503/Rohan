/*! \mainpage Rohan Multivalued Neural Network Simulator
 *
 * \section intro_sec Introduction
 *
 * Rohan was developed by Jeff Wilson <jwilson@clueland.com> at the Texas A&M University - Texarkana Computational Intelligence Laboratory < http://www.tamut.edu/CIL/ > under the direction of Dr Igor Aizenberg.
 *
 * Funded by National Science Foundation grant #0925080
 *
 * \section install_sec Installation
 *
 * \subsection step1 Step 1: Opening the box
 *  
 * Hopefully included in the 1.0 or later release.
 */
// Rohan.cpp : Defines the entry point for the console application.

//warning tamers

//#define sprintf "sprintf_s"

/* Includes, cuda */
#include "Rohan.h"
#include "Rohan-data.h"
#include "Rohan-io.h"
#include "Rohan-learn.h"

#include "Rohan-kernel.h"
#include "cBarge.h"
#include "cDrover.h"
#include "cTeam.h"
#include "ShowMe.h"
#include "stdafx.h"
//#include <time.h> // for tsrtuct
#include <sys/timeb.h>
#include <stdlib.h>
using namespace std;
using std::cin;
using std::cout;
#include <math.h>  //for sqrt()
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))
#define TWO_PI 6.283185307179586476925286766558
/// globals
int iDebugLvl;
int iTrace;
__device__ struct rohanContext devSes;
__device__ struct rohanLearningSet devLearn;
__device__ struct rohanNetwork devNet;

//boost test
//#include <boost/lambda/lambda.hpp>
//#include <iostream>
//#include <iterator>
//#include <algorithm>

int _tmain(int argc, _TCHAR* argv[])
{mIDfunc/// general program procedure is to setup preparations for the duty loop, execute it, then do housekeeping after
	cutilSafeCall( cudaSetDevice(0) ); /// all cuda calls to run on headless first device
	iDebugLvl=1;
	
	struct rohanContext rSes /* This is the master session context object, with pointers to the learning set and network objects as members */;
	struct rohanNetwork rNet;
	struct rohanLearningSet rLearn;

	// pointers to learning set and network object copies in both cpu and gpu memory spaces are recorded
	rSes.rLearn = &rLearn;
	rSes.rNet = &rNet; 
	cudaGetSymbolAddress( (void**)&rSes.devLearn, "devLearn" );
		mCheckCudaWorked
	cudaGetSymbolAddress( (void**)&rSes.devNet, "devNet" );
		mCheckCudaWorked
	cudaGetSymbolAddress( (void**)&rSes.devSes, "devSes" );
		mCheckCudaWorked

	//establish debugging file streams
	//AsciiFileHandleWrite("debug.txt", &rSes.debugHandle);
	//AsciiFileHandleWrite("bitbucket.txt", &rSes.bitBucket);
	//fprintf(rSes.debugHandle, "-> sizeof rSes %d, rLearn %d, rNet %d <-\n", sizeof(rSes), sizeof(rLearn), sizeof(rNet));
	//fprintf(rSes.debugHandle, "-> sizeof short %d, int %d, long %d, float %d, double %d, cdc %d <-\n", sizeof(short), sizeof(int), sizeof(long), sizeof(float), sizeof(double), sizeof(cuDoubleComplex));
	//printf("-> sizeof rSes %d, rLearn %d, rNet %d <-\n", sizeof(rSes), sizeof(rLearn), sizeof(rNet));
	//for(int i=0; (i+10)<5; ++i)printf("i%d\t", i);printf("\n");
	//printf("-> sizeof short %d, int %d, long %d, float %d, double %d, cdc %d, * cdc %d, FILE* %d, void* %d, cudaDeviceProp %d <-\n", 
	//	sizeof(short), sizeof(int), sizeof(long), sizeof(float), sizeof(double), sizeof(cuDoubleComplex), sizeof(cuDoubleComplex*), sizeof(FILE*), sizeof(void*), sizeof(cudaDeviceProp) );
	
	// create class objects
	cDeviceTeam cdtHorse(rSes); // the horse handles GPU computation kernels and their currency
	cBarge cbBarge(rSes); // the barge loads and holds common data like the learning set and weights
	cDrover cdDrover(rSes); // the drover handles the user interface and directs the other objects

	// proceed with operations
	if(cdDrover.DoAnteLoop(argc, argv, &cdDrover, &cbBarge, &cdtHorse)){
		cdDrover.DoMainLoop(rSes);
	}
	cdDrover.DoPostLoop(rSes);
	//fclose(rSes.debugHandle);

	// end of operations
	exit (0);
}

