#include "Rohan.h"
#include "Rohan-data.h"
#include "Rohan-io.h"
#include "Rohan-learn.h"
#include "Rohan-menu.h"
#include "Rohan-kernel.h"
#include "ShowMe.h"
#include "stdafx.h"
#include <conio.h> //for _getch 
#include "cTeam.h"
#include "cBarge.h"
#include "cDrover.h"
#include <cuda.h>
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <multithreading.h>

//#include <time.h> // for tsrtuct
#include <sys/timeb.h>

#include <stdlib.h>
using namespace std;
using std::cin;
using std::cout;

#define TWO_PI 6.283185307179586476925286766558
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))

extern int iDebugLvl, iWarnings, iErrors, iTrace;
extern long bCUDAavailable;


//////////////// class cBarge begins ////////////////

void cBarge::ShowMe()
{
	//ShowMeSes(* rSes, false);
	printf("Am stout barje.\n");
}


long cBarge::SetContext( rohanContext& rC)
{/// enables pointer access to master context struct
	rSes = &rC;
	rLearn = rC.rLearn;
	rNet = rC.rNet;
	return 0;
}


long cBarge::SetDrover( class cDrover * cdDrover)
{/// enables pointer access to active Drover object
	Drover = cdDrover;
	return 0;
}


long cBarge::SetTeam( class cDeviceTeam * cdtTeam)
{/// enables pointer access to active Team object
	Team = cdtTeam;
	return 0;
}


long cBarge::ObtainSampleSet(struct rohanContext& rSes)
{mIDfunc /// loads the learning set to be worked with Ante-Loop
	long iReturn=0; 
	
	FILE *fileInput;
	// File handle for input
	
	iReturn=AsciiFileHandleRead(rSes.rLearn->sLearnSet, &fileInput);
	if (iReturn==0) // unable to open file
		++rSes.iErrors;
	else{ // file opened normally
		// file opening and reading are separated to allow for streams to be added later
		long lLinesRead=DoLoadSampleSet(rSes, fileInput);
		if (lLinesRead) {
			printf("Parsed %d lines from %s\nStored %d samples, %d input values, %d output values each.\n", 
				lLinesRead, rSes.rLearn->sLearnSet, rSes.rLearn->lSampleQty, rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
			//verify samples fall wihtin sector values
			if(CurateSectorValue(rSes)) {
				CompleteHostLearningSet(rSes);
				//if(rSes.bCUDAavailable){
					//load samples into the parallel structures in the GPU memory
					//devCopySampleSet(rSes);
					//load complex samples into the parallel structures in the host memory
					//LetCplxCopySamples(rSes);
				//}
			}
			else{
				return 0;
			} //endif for CurateSectorValue
		}
		else {
			printf("No Samples Read by cuLoadSampleSet\n");
			iReturn=0;
		}
	}

	return iReturn;
}


long cBarge::DoLoadSampleSet(struct rohanContext& rSes, FILE *fileInput)
{mIDfunc	/// pulls in values from .txt files, used for testing before main loop
	// Return:    long
	//	0 = error
	//FILE *fShow=rSes.bitBucket;

	#define MAX_REC_LEN 65536 /* Maximum size of input buffer */
	//fprintf(fShow, "%s %d:\n", __FILE__, __LINE__);

	long  lLinesQty=0,lMaxLines=256; /* countable number of lines, number of lines with memory allocation */
	char  cThisLine[MAX_REC_LEN]; /* Contents of current line */
	int iArchLineIdx; // The line number that has the sample qty and topology and file params in it
	rSes.rLearn->iValuesPerLine=0; rSes.rLearn->lSampleQty=0;
	// reset quantities for counting later
	char **cLines = (char **)malloc(256 * sizeof (char *));
	// 1d array of lines in text file beginning with first line in position zero
	// learning set format allows for first or first and second lines to he paramteres rather than samples

	while (fgets(cThisLine, MAX_REC_LEN, fileInput)) { //each line is read in turn
		//fprintf(fShow, "|%s", cThisLine);
		cLines[lLinesQty++] = _strdup(cThisLine); // each line is copied to a string in the array
		if (!(lMaxLines > lLinesQty)) {  // if alloated space is used up, double it.
			lMaxLines *= 2;
			void * temp = realloc(cLines, lMaxLines * sizeof (char *));
			if (!temp) {
				  for (int k=0;k<lLinesQty;++k) {
					  free(cLines[k]);
				  }
				  printf("Realloc ran out of space?  OH NOES! %s line %d\n", __FILE__, __LINE__);
				  return 0;
			} else {
				  cLines = (char **)temp;
			}
		}
	}
	fclose(fileInput); // close stream when fgets returns false (no more lines)
	//fprintf(fShow, "%s %d: %d lines read.\n", __FILE__, __LINE__, lLinesQty);
	// this should be a shrinking, and should never fail.
	cLines = (char **)realloc(cLines, lLinesQty * sizeof (char*));
		mCheckMallocWorked(cLines)
	// 1 means ContActivation
	// 0 means discrete, and values of 2+ indicate parameter has been omitted,
	// defaulting to discrete, and value is actually # of samples in file
	rSes.rLearn->iEvalMode=atof(cLines[0])>1 ? 1 : 0; iArchLineIdx = 1 - rSes.rLearn->iEvalMode;
		if (rSes.rLearn->iEvalMode) printf ("Discrete output values indicated.\n");
	else printf("Continuous output values indicated.\n");
		//mDebug(1, 0) printf("\"%s\"%siEvalMode %d, iArchLineIdx %d\n", cLines[0], cLines[1], rSes.rLearn->iEvalMode, iArchLineIdx);

	char *sArch, *tok;

	sArch = _strdup(cLines[iArchLineIdx]); // make a sacrificial copy
	tok = strtok(sArch, " ,\t");
	if (sArch==NULL) // no additional params present
		printf("No params present; fetch from config file or command line (not yet supported XX).\n");
	else {
		cuMakeArchValues(cLines[iArchLineIdx], rSes);
		rSes.rLearn->iInputQty=rSes.rNet->rLayer[0].iNeuronQty;
		rSes.rLearn->iOutputQty=rSes.rNet->rLayer[rSes.rNet->iLayerQty-1].iNeuronQty;
		printf("%d inputs, %d output(s) specified.\n", rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
		//fprintf(fShow, "%s %d: %d inputs, %d output(s) specified.\n", __FILE__, __LINE__, rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
	}
	
	rSes.rLearn->lSampleQty=(long)atof(cLines[iArchLineIdx]);
	printf("%d samples specified: ", rSes.rLearn->lSampleQty);
			
// Parses lines of text for input values and output value and stores them in dynamic int arrays
// returns # of inputs per line
	char *pch; 
	char  *cSample;

	long lCurrentLine=atof(cLines[0])>1 ? 1 : 2; //find which line the samples begin
	cSample=_strdup(cLines[2]); // strtok chops up the input string, so we must make a copy
	pch = strtok (cSample, " ,\t");
	while (pch != NULL) {// this loop counts the values present in a copy of line 2, which has to be a sample line
		pch = strtok (NULL, " ,\t"); ++rSes.rLearn->iValuesPerLine;
	}
		//mDebug(1,0) printf("%d inputs, %d outputs, %d values per line.\n",rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty, rSes.rLearn->iValuesPerLine);
		//printf("%d inputs, %d output(s) specified.\n", rSes.rLearn->iValuesPerLine - rSes.rLearn->iOutputQty, rSes.rLearn->iOutputQty);
	int iExcessValueQty = rSes.rLearn->iValuesPerLine - (rSes.rLearn->iInputQty + rSes.rLearn->iOutputQty );
	if(iExcessValueQty>0) {
		fprintf(stderr, "Warning: %d unused values in sample tuples.\n", iExcessValueQty);
		++rSes.iWarnings;
	}
	if(iExcessValueQty<0) {
		fprintf(stderr, "Error: %d values not found in sample tuples.\n", iExcessValueQty*-1);
		++rSes.iErrors;
	}
	
	/// allocate memory for tuple storage
	rSes.rLearn->dXInputs = (double*)malloc( (rSes.rLearn->iInputQty+1) * rSes.rLearn->lSampleQty * sizeof(double)); // scalar X input signal
		mCheckMallocWorked(rSes.rLearn->dXInputs)
	rSes.rLearn->dDOutputs=(double*)malloc( (rSes.rLearn->iOutputQty+1) * rSes.rLearn->lSampleQty * sizeof(double)); // scalar D correct output signal
		mCheckMallocWorked(rSes.rLearn->dDOutputs)
	
	for (long s=0; s<rSes.rLearn->lSampleQty; ++s){ //iterate over the number of samples and malloc
		for (int k=0; k<=rSes.rLearn->iInputQty; ++k) // fill with uniform, bogus values
			//#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))
			rSes.rLearn->dXInputs[ IDX2C( k, s, rSes.rLearn->iInputQty+1 ) ]=-999.9;
		for (int k=0; k<=rSes.rLearn->iOutputQty; ++k) // fill with uniform, bogus values
			rSes.rLearn->dDOutputs[ IDX2C( k, s, rSes.rLearn->iOutputQty+1 ) ]=-888.8; 
		// parse and store sample values
		pch = strtok (cLines[lCurrentLine], " ,\t"); // get the first token on a line
		//fprintf(fShow, "%d: ", s);
		//if(rSes.bRInJMode){ // if flag for compatibility with older NN simulator is set XX
		//	for (int k=rSes.rLearn->iValuesPerLine; k>=1; --k){ // save it beginning with last position
		//		rSes.rLearn->dXInputs[ IDX2C( s, k, rSes.rLearn->iInputQty+1 ) ]=atof(pch); // convert and assign each value in a line
		//		pch = strtok (NULL, " ,\t");
		//	}
		//}
		//else{ // otherwise store things the usual way
			for (int k=1; k<=rSes.rLearn->iInputQty; ++k){ // save it beginning with position 1
				rSes.rLearn->dXInputs[ IDX2C( k, s, rSes.rLearn->iInputQty+1 ) ]=atof(pch); // convert and assign each value in a line
				//fprintf(fShow, "%s,%f,%d,%d\t", pch, atof(pch), k, IDX2C( k, s, rSes.rLearn->iInputQty+1) );
				pch = strtok (NULL, " ,\t");
			}
		//}
		//fprintf(fShow, "\n%d: ", s);
		for (int k=1; k<=rSes.rLearn->iOutputQty; ++k){
			rSes.rLearn->dDOutputs[ IDX2C( k, s, rSes.rLearn->iOutputQty+1 ) ]=atof(pch); // convert and assign each value in a line
			//fprintf(fShow, "%s,%f,%d,%d\t", pch, atof(pch), k, IDX2C( k, s, rSes.rLearn->iOutputQty+1) );
			pch = strtok (NULL, " ,\t");
		}
		//fprintf(fShow, "\n");
		rSes.rLearn->dXInputs[ IDX2C( 0, s, rSes.rLearn->iInputQty+1 ) ]=0.0; // virtual input zero should always be zero
		rSes.rLearn->dDOutputs[ IDX2C( 0, s, rSes.rLearn->iOutputQty+1) ]=0.0; // output neuron zero should always produce sector 0 output
		free(cLines[lCurrentLine]); //if (lCurrentLine<10) printf("freed cLine[%d]\n", lCurrentLine);
		++lCurrentLine;
	}
	
	free(cLines[0]);
	if (iArchLineIdx) free(cLines[1]); // WEIRD MEMORY ERRORS? LOOK HERE XX
	// above line avoids double-freeing cLines[1] if it was used for a sample instead of the sample qty
	free(cLines);
 	return lLinesQty; // returns qty of lines read from file, not the same as quantity of samples
}


long cBarge::CurateSectorValue(struct rohanContext& rSes)
{mIDfunc /// compares sector qty to sample values for adequate magnitude
	int iOverK=0;
	//FILE *fShow=rSes.bitBucket;
	
	// debug header
	//fprintf( fShow, "%s %d: %d inputs, %d outputs, %d sectors.\n" , __FILE__, __LINE__, rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty, rSes.rNet->iSectorQty );
	//fprintf( fShow, "%s %d: working on %d lines.\nCurating:\n" , __FILE__, __LINE__, rSes.rLearn->lSampleQty );
	// loop over samples for inputs
	for (long s=0; s<rSes.rLearn->lSampleQty; ++s){
		//fprintf( fShow, "%dX|", s);
		for (int i=0; i<=rSes.rLearn->iInputQty; ++i){
			if(rSes.rLearn->dXInputs[ IDX2C( i, s, rSes.rLearn->iInputQty+1 )]>=rSes.rNet->iSectorQty){
				// if any input values fall beyond the maximum sector value, alert and make recommendation
				fprintf(stderr, "Error: Sample #%d has value that exceeds sector qty %d; suggest increasing to %d!\n",
					s, rSes.rNet->iSectorQty, static_cast<int>(floor(rSes.rLearn->dXInputs[ IDX2C( i, s, rSes.rLearn->iInputQty+1 ) ]*1.33)+1));
				++iOverK;
			}
			//fprintf( fShow, "%.6g,%d\t", rSes.rLearn->dXInputs[ IDX2C( i, s, rSes.rLearn->iInputQty+1 ) ], i );
		}	
		//fprintf( fShow, "\n%dD|", s);
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ // now loop over output values
			if(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 )]>=rSes.rNet->iSectorQty){
				// if any output values fall beyond the maximum sector value, alert and make recommendation
				fprintf(stderr, "Error: Sample #%d has value that exceeds sector qty %d; suggest increasing to %d!\n",
					s, rSes.rNet->iSectorQty, static_cast<int>(floor(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ]*1.33)+1));
				++iOverK;
			}
			//fprintf( fShow, "%.6g,%d\t", rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ], i );
		}
		//fprintf( fShow, "\n");
	}
	if (iOverK)	{ // any out-of-bounds values are a fatal error
		++rSes.iErrors;
		return 0;
	}

	return rSes.rLearn->lSampleQty; // return number of samples veified within parameters
}


long cBarge::CompleteHostLearningSet(struct rohanContext& rSes)
{mIDfunc //allocate and fill arrays of complx values converted from scalar samples, all in host memory
	long iReturn=0;
	long IQTY, OQTY, INSIZED, OUTSIZED, INSIZECX, OUTSIZECX;
	
	//setup dimension values
	IQTY = rSes.rLearn->iInputQty+1 ;
	INSIZED = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(double) ;
	INSIZECX = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(cuDoubleComplex) ;
	OQTY = rSes.rLearn->iOutputQty+1; 
	OUTSIZED = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(double);
	OUTSIZECX = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(cuDoubleComplex);

	// allocate remainder of host scalar arrays
	rSes.rLearn->dYEval=(double*)malloc( OUTSIZED ); // scalar Y evaluated output signal
		mCheckMallocWorked(rSes.rLearn->dYEval)
	rSes.rLearn->dAltYEval=(double*)malloc( OUTSIZED ); // alt method scalar output
		mCheckMallocWorked(rSes.rLearn->dAltYEval)
	rSes.rLearn->dSE1024=(double*)malloc( OUTSIZED ); // array for RMSE calculation, changed to OUTSIZED 1/8/12
		mCheckMallocWorked(rSes.rLearn->dSE1024)
	// allocate host complex arrays
	rSes.rLearn-> cdcXInputs  =(cuDoubleComplex*)malloc( INSIZECX ); // cx X Input signal
		mCheckMallocWorked(rSes.rLearn->cdcXInputs)
	rSes.rLearn-> cdcDOutputs =(cuDoubleComplex*)malloc( OUTSIZECX ); // cx D desired output signal
		mCheckMallocWorked(rSes.rLearn->cdcDOutputs)
	rSes.rLearn-> cdcYEval    =(cuDoubleComplex*)malloc( OUTSIZECX ); // cx evaluated Y output signal
		mCheckMallocWorked(rSes.rLearn->cdcYEval)
	rSes.rLearn-> cdcAltYEval =(cuDoubleComplex*)malloc( OUTSIZECX ); // cx alt evaluation Y output
		mCheckMallocWorked(rSes.rLearn->cdcAltYEval)

	for(long S=0;S<rSes.rLearn->lSampleQty; S++){
		for (int I=0;I< IQTY ; I++){
			rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )]
				= ConvScalarCx( rSes, rSes.rLearn-> dXInputs [IDX2C( I, S, IQTY )] ); // convert scalar inputs on host
		}
		for (int O=0;O< OQTY ; O++){
			rSes.rLearn-> dYEval      [IDX2C(  O, S, OQTY )] = S; 
			rSes.rLearn-> dAltYEval	  [IDX2C(  O, S, OQTY )] = -S;
			rSes.rLearn-> dSE1024	  [IDX2C(  O, S, OQTY )] = O;
			rSes.rLearn-> cdcDOutputs [IDX2C(  O, S, OQTY )]
				= ConvScalarCx( rSes, rSes.rLearn->dDOutputs[IDX2C(  O, S, OQTY )] ); // convert cx desired outputs
			rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].x = O; 
			rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].y = S; 
			rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].x = -1*O;
			rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].y = -1*S;
		}
	}
	
	return iReturn;
}


//long cBarge::LetCplxCopySamples(struct rohanContext& rSes)
//{mIDfunc //load complex samples into the parallel structures in the host memory
//	long iReturn=0;
//
//	rSes.rLearn->gpuXInputs=(cuDoubleComplex*)malloc( rSes.rLearn->lSampleQty * (rSes.rLearn->iInputQty+1) * sizeof(cuDoubleComplex) ); // allocate block of cx inputs
//		mCheckMallocWorked(rSes.rLearn->gpuXInputs)
//			printf("%d x %d block allocated\n", rSes.rLearn->lSampleQty, rSes.rLearn->iInputQty+1 ); 
//	rSes.rLearn->gpuDOutputs=(cuDoubleComplex*)malloc( rSes.rLearn->lSampleQty * (rSes.rLearn->iOutputQty+1) * sizeof(cuDoubleComplex) ); // allocate block of cx desired outputs
//		mCheckMallocWorked(rSes.rLearn->gpuDOutputs)
//			printf("%d x %d block allocated, %d locations\n", rSes.rLearn->lSampleQty, rSes.rLearn->iOutputQty+1, rSes.rLearn->lSampleQty * (rSes.rLearn->iOutputQty+1) ); 
//	rSes.rLearn->gpuOutScalar=(cuDoubleComplex*)malloc( rSes.rLearn->lSampleQty * (rSes.rLearn->iOutputQty+1) * sizeof(cuDoubleComplex) ); // allocate block of desired and yielded scalars
//		mCheckMallocWorked(rSes.rLearn->gpuOutScalar)
//			printf("%d x %d block allocated\n", rSes.rLearn->lSampleQty, rSes.rLearn->iOutputQty+1 ); 
//	
//	for(long S=0;S<rSes.rLearn->lSampleQty; S++){
//		for (int I=0;I<rSes.rLearn->iInputQty+1; I++){
//			rSes.rLearn->gpuXInputs[IDX2C( S, I, rSes.rLearn->iInputQty+1)] = 
//				rSes.rLearn->cdcXInputs[IDX2C( S, I, rSes.rLearn->iInputQty+1)]; // copy cx inputs
//		}
//		for (int O=0;O<rSes.rLearn->iOutputQty+1; O++){
//			rSes.rLearn->gpuDOutputs[IDX2C( S, O, rSes.rLearn->iOutputQty+1)] = 
//				rSes.rLearn->cdcDOutputs[IDX2C( S, O, rSes.rLearn->iOutputQty+1)]; // copy cx desired outputs
//			//printf("Location %d stored\n", IDX2C(O,S, rSes.rLearn->iOutputQty));
//			rSes.rLearn->gpuOutScalar[IDX2C( S, O, rSes.rLearn->iOutputQty+1)].x	= 
//				rSes.rLearn->dDOutputs[IDX2C( S, O, rSes.rLearn->iOutputQty+1)]; // copy desired and yielded scalars
//		}
//	}
//
//	return iReturn;
//}		


long cBarge::DoPrepareNetwork(struct rohanContext& rSes)
{mIDfunc /// sets up network poperties and data structures for use
	int iReturn=0;
	// on with it
	
	cuMakeNNStructures(rSes); // allocates memory and populates network structural arrays
	iReturn=BinaryFileHandleRead(rSes.rNet->sWeightSet, &rSes.rNet->fileInput);
	// file opening and reading are separated to allow for streams to be added later
	if (iReturn) {
		long lWeightsRead=cuNNLoadWeights(rSes, rSes.rNet->fileInput);
			if (lWeightsRead) printf("Parsed and assigned %d complex weights from %s\n", lWeightsRead, rSes.rNet->sWeightSet);
			else {
				fprintf(stderr, "Error: No Weights Read by cuNNLoadWeights\n");
				++rSes.iErrors;
				//printf("Waiting on keystroke...\n"); _getch(); return iReturn;
			}
	}
	else { // can't open, user random weights
		printf("Can't open %s, using random weights.\n", rSes.rNet->sWeightSet);
		cuRandomizeWeights(rSes); // populate network with random weight values
	}


// record fixed-length accumulation 
	struct rohanNetwork * rnSrc; //, * rnDest ;
	struct rohanLayer * rlSrc;
	long LQTY, LLAST, LSIZE; //, SECSIZE;
	rnSrc=(rSes.rNet);
	long SIZE = sizeof(*rSes.rNet);
	
	LQTY = rnSrc->iLayerQty ; 
		LLAST = LQTY - 1 ;
	LSIZE = sizeof(rohanLayer) * LQTY ;
	//kind=cudaMemcpyHostToDevice;
	
	long DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE;
	NQTY = rnSrc->rLayer[0].iNeuronQty + 1 ; // neurons = outgoing signals
	NSIZE = NQTY * sizeof(cuDoubleComplex) ;
	rlSrc=&(rSes.rNet->rLayer[0]);
	
	// blank original values
	for (long i = 0; i<MAXWEIGHTS; ++i) rSes.rNet->Wt[i]=cdcZero;
	for (long i = 0; i<MAXNEURONS; ++i) rSes.rNet->Deltas[i]=rSes.rNet->Signals[i]=cdcZero;
	rSes.rNet->iNeuronQTY[0]=rSes.rLearn->iInputQty+1; // initialize with inputs for Layer Zero
	rSes.rNet->iDendrtOfst[0]=rSes.rNet->iDendrtQTY[0]=rSes.rNet->iNeuronOfst[0]=rSes.rNet->iWeightOfst[0]=rSes.rNet->iWeightQTY[0]=0;

	//printf("layer %d molded and filled?\n", 0);

	//for (long L=1; L<=LLAST; ++L){
	for (long L=1; L<MAXLAYERS; ++L){
		if (L<=LLAST){
			//setup dimension values
			DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = weights
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
			rlSrc=&(rSes.rNet->rLayer[L]);
		}else{
			// fill out unused layers with empty structures
			DQTY=rSes.rNet->iNeuronQTY[L-1]; //	dendrites = previous layer's neurons
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY=1; // neuonrs limited to Nzero
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = neurons * dendrites
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
		}
	
// track fixed-length accumulation - should move to net setup
		//rSes.rNet->iDendrtOfst[L]=rSes.rNet->iDendrtOfst[L-1]+rSes.rNet->iDendrtQTY[L-1];
		rSes.rNet->iNeuronOfst[L]=rSes.rNet->iNeuronOfst[L-1]+rSes.rNet->iNeuronQTY[L-1];
		rSes.rNet->iWeightOfst[L]=rSes.rNet->iWeightOfst[L-1]+rSes.rNet->iWeightQTY[L-1];
		rSes.rNet->iDendrtQTY[L]=DQTY;
		rSes.rNet->iNeuronQTY[L]=NQTY;
		rSes.rNet->iWeightQTY[L]=WQTY;
		if(rSes.rNet->iWeightOfst[L]+rSes.rNet->iWeightQTY[L] > MAXWEIGHTS){
			++rSes.iErrors;
			fprintf(stderr, "MAXIMUM WEIGHTS EXCEEDED at layer %d!\n", L);
		}
		if(rSes.rNet->iNeuronOfst[L]+rSes.rNet->iNeuronQTY[L] > MAXNEURONS){
			++rSes.iErrors;
			fprintf(stderr, "MAXIMUM NEURONS EXCEEDED at layer %d!\n", L);
		}
		// copy each layer's weights into the fixed-length array beginning at an offset
		if (L<=LLAST){
			for(int i=0; i<WQTY; ++i)
				rSes.rNet->Wt[i+rSes.rNet->iWeightOfst[L]]=rlSrc->Weights[i];
		}else{
			for(int i=1; i<WQTY; ++i)
				rSes.rNet->Wt[i+rSes.rNet->iWeightOfst[L]]=cdcZero;
			rSes.rNet->Wt[0+rSes.rNet->iWeightOfst[L]]=cdcIdentity;
		}
//		printf("all:%d= %08lX, L%d:%d= %08lX\n", MAXWEIGHTS, crc32buf((char*)&rSes.rNet->Wt, MAXWEIGHTS * 16), 
//			L, rSes.rNet->iWeightQTY[L], crc32buf((char*)&rSes.rNet->Wt[0+rSes.rNet->iWeightOfst[L]], rSes.rNet->iWeightQTY[L] * 16) );
	}
	
	//for(int j=0; j<MAXLAYERS; ++j){
	//	printf("Layer %d, Nq %d Dq %d Wq %d: Nof %d Wof %d\n", j, rSes.rNet->iNeuronQTY[j], rSes.rNet->iDendrtQTY[j], rSes.rNet->iWeightQTY[j], rSes.rNet->iNeuronOfst[j], rSes.rNet->iWeightOfst[j] );
	//}
	//for (int j=1; j < rSes.rNet->iLayerQty; ++j){ //no weights for layer 0
	//	struct rohanLayer& lay = rSes.rNet->rLayer[j];
	//	for (int k=1; k <= lay.iNeuronQty; ++k){ // no weights for neuron 0
	//		for (int i=0; i <= lay.iDendriteQty; ++i){
	//			cuDoubleComplex way = rSes.rNet->Wt[IDX2C(i+rSes.rNet->iWeightOfst[j], k, rSes.rNet->iDendrtQTY[j])] ;
	//			cuDoubleComplex oway = rSes.rNet->rLayer[j].Weights[IDX2C( i, k, rSes.rNet->rLayer[j].iDendriteQty+1 )] ;
	//			printf("L%d N%d D%d %f+%f\n", j, k, i, way.x-oway.x, way.y-oway.y);
	//		}
	//	}
	//}

	// store pointers to dev global structures
	cudaGetSymbolAddress( (void**)&rSes.rNet->gWt, "devNet");
		mCheckCudaWorked
	rSes.rNet->gDeltas=rSes.rNet->gSignals=rSes.rNet->gWt;
	rSes.rNet->gDeltas += offsetof(rohanNetwork, Deltas);
	rSes.rNet->gSignals += offsetof(rohanNetwork, Signals);
	rSes.rNet->gWt += offsetof(rohanNetwork, Wt);
	iReturn=cuSectorTableMake(rSes); // fill the table with values
	if (iReturn==0) {
		printf("Out of Memory in cuSectorTableMake\n");
		++rSes.iErrors;
		//printf("Waiting on keystroke...\n");
		//_getch();
	}



//printf("Dall> %08lX\n", crc32buf((char*)(rSes.rNet->Deltas), MAXNEURONS * 16) );
//printf("Sall> %08lX\n", crc32buf((char*)(rSes.rNet->Signals), MAXNEURONS * 16) );
//printf("Wall> %08lX\n", crc32buf((char*)(rSes.rNet->Wt), MAXWEIGHTS * 16) );
//printf("D1>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Deltas[0+rSes.rNet->iNeuronOfst[1]], rSes.rNet->iNeuronQTY[1] * 16) , rSes.rNet->iNeuronOfst[1] , rSes.rNet->iNeuronQTY[1] );
//printf("D2>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Deltas[0+rSes.rNet->iNeuronOfst[2]], rSes.rNet->iNeuronQTY[2] * 16) , rSes.rNet->iNeuronOfst[2] , rSes.rNet->iNeuronQTY[2] );
//printf("S1>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Signals[0+rSes.rNet->iNeuronOfst[1]], rSes.rNet->iNeuronQTY[1] * 16) , rSes.rNet->iNeuronOfst[1] , rSes.rNet->iNeuronQTY[1] );
//printf("S2>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Signals[0+rSes.rNet->iNeuronOfst[2]], rSes.rNet->iNeuronQTY[2] * 16) , rSes.rNet->iNeuronOfst[2] , rSes.rNet->iNeuronQTY[2] );
//printf("W1>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Wt[0+rSes.rNet->iWeightOfst[1]], rSes.rNet->iWeightQTY[1] * 16) , rSes.rNet->iWeightOfst[1] , rSes.rNet->iWeightQTY[1] );
//printf("W2>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Wt[0+rSes.rNet->iWeightOfst[2]], rSes.rNet->iWeightQTY[2] * 16) , rSes.rNet->iWeightOfst[2] , rSes.rNet->iWeightQTY[2] );




	return iReturn;
}


long cBarge::LetWriteWeights(struct rohanContext& rSes)
{mIDfunc/// saves weight values to disk
	long lReturn;
	// dump weights for verification
	FILE *fileOutput; // File handle for output
	lReturn=AsciiFileHandleWrite("weightdump.txt", &fileOutput);
	AsciiWeightDump(rSes, fileOutput); 
	return lReturn;
}

long cBarge::LetWriteEvals(struct rohanContext& rSes, struct rohanLearningSet& rLearn)
{mIDfunc/// saves evaluated output values to disk
	long lReturn;
	FILE *fileOutput; // File handle for output
	char sFileAscii[255]; //="DefaultSession";

	strncpy(sFileAscii,rSes.sSesName,246); // do not exceed 254 char file name
	strcat(sFileAscii,"Evals.txt");
	lReturn=AsciiFileHandleWrite(sFileAscii, &fileOutput);
	if(lReturn){
		for(long s=0; s<rSes.lSampleQtyReq; ++s){
			if(rSes.iSaveInputs){
				for(int i=1; i<=rLearn.iInputQty; ++i) // write inputs first
					fprintf(fileOutput, "%3.f, ", rLearn.dXInputs[IDX2C(i,s,rLearn.iInputQty+1)]);
			}
			if(rSes.iSaveOutputs){
				for(int i=1; i<=rLearn.iOutputQty; ++i) // write desired outputs second
					fprintf(fileOutput, "%7.3f, ", rLearn.dDOutputs[IDX2C(i,s,rLearn.iOutputQty+1)]);
			}
			for(int i=1; i<=rLearn.iOutputQty; ++i){ // write yielded outputs third
				fprintf(fileOutput, "%#7.3f", rLearn.dYEval[IDX2C(i,s,rLearn.iOutputQty+1)]);
				if (i<rLearn.iOutputQty)
					fprintf(fileOutput, ", "); // only put commas between outputs, not after
			}
			if(rSes.iSaveSampleIndex){ // write sample indexes last
				fprintf(fileOutput, ", %d", s);
			}
			fprintf(fileOutput, "\n"); // end each line with a newline
		}
		fclose(fileOutput);
		printf("%d evals writen to %s", (lReturn=rSes.lSampleQtyReq), sFileAscii ); // document success and filename
	}

	return lReturn; // reutrn number of sample evals recorded
}

long cBarge::ShowDiagnostics()
{mIDfunc
	
	printf("cBarge diagnostics: ");
	if(Team==NULL)
		printf("No team!\n");
	if(Drover==NULL)
		printf("No drover!\n");
	if(rSes->rLearn->cdcXInputs==NULL)
		printf("No complex inputs at host!\n");
	if(rSes->rLearn->dDOutputs==NULL)
		printf("No scalar outputs at host!\n");
	if(rSes->rLearn->cdcDOutputs==NULL)
		printf("No complex outputs at host!\n");
	if (rLearn==NULL)
		printf("No rLearn structure!\n");
	else
		//printf("Holding %d samples w/ %d inputs, %d output(s)\n", *rLearn->lSampleQty, *rLearn->iInputQty, *rLearn->iOutputQty);
		printf("Barge is holding %d samples w/ %d inputs, %d output(s).\n", rSes->rLearn->lSampleQty, rSes->rLearn->iInputQty, rSes->rLearn->iOutputQty);
	
	return 0;
}


long cBarge::DoCuFree(struct rohanContext &rSes)
{mIDfunc/// free allocated memory for all structures
	
	cuFreeNNTop(rSes); // free network topology structures
	cuFreeLearnSet(rSes); // free learning set structures
	
	return 0;
}


long cBarge::cuFreeNNTop(struct rohanContext &rSes)
{mIDfunc/// frees data structures related to network topology
	//cublasStatus csStatus;
	
	free( rSes.rNet->cdcSectorBdry );
	// layer components
	free( rSes.rNet->rLayer[0].ZOutputs ); // Layer Zero has no need of weights!
	//csStatus = cublasFree( rSes.rNet->rLayer[0].ZOutputs ); // de-allocate a GPU-space pointer to a vector of complex neuron outputs for each layer
	//mCuMsg(csStatus,"cublasFree()")
	
	for (int i=1; i < rSes.rNet->iLayerQty; ++i){ 
		struct rohanLayer& lay=rSes.rNet->rLayer[i];
		
		free( lay.Weights ); // free the weights
		free( lay.Deltas ); // free the backprop areas
		free( lay.XInputs ); // free the inputs
		free( lay.ZOutputs ); // free the outputs
	}
	free( rSes.rNet->rLayer ); // free empty layers
	printf("Network structures freed.\n");
	return 0;
}


long cBarge::cuFreeLearnSet(struct rohanContext &rSes)
{mIDfunc/// free the learning set of samples
	
	free( rSes.rLearn->dXInputs ); 
	free( rSes.rLearn->dDOutputs );
	free( rSes.rLearn->dYEval ); 
	free( rSes.rLearn->dAltYEval ); 
	free( rSes.rLearn->dSE1024 );
	
	free( rSes.rLearn->cdcXInputs ); 
	free( rSes.rLearn->cdcDOutputs ); 
	free( rSes.rLearn->cdcYEval ); 
	free( rSes.rLearn->cdcAltYEval );

	
	printf("Learning set structures freed.\n");

	return 0;
}


