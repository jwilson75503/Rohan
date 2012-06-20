/* Includes, cuda */
#include "stdafx.h"

extern int gDebugLvl, gTrace;
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
	//rSes.cbBarge=this;
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
			}
			else{
				return 0;
			} 
		}
		else {
			printf("No Samples Read by cuLoadSampleSet\n");
			iReturn=0;
		}
	}

	return iReturn;
}


long cBarge::DoLoadSampleSet(struct rohanContext& rSes, FILE *fileInput)
{mIDfunc/// pulls in values from .txt files, used for testing before main loop
	// Return:    long
	//	0 = error

	#define MAX_REC_LEN 65536 /* Maximum size of input buffer */

	long  lLinesQty=0,lMaxLines=256; /* countable number of lines, number of lines with memory allocation */
	char  cThisLine[MAX_REC_LEN]; /* Contents of current line */
	int iArchLineIdx; // The line number that has the sample qty and topology and file params in it
	rSes.rLearn->iValuesPerLine=0; rSes.rLearn->lSampleQty=0;
	// reset quantities for counting later
	char **cLines = (char **)malloc(256 * sizeof (char *));
	// 1d array of lines in text file beginning with first line in position zero
	// learning set format allows for first or first and second lines to he paramteres rather than samples

	while (fgets(cThisLine, MAX_REC_LEN, fileInput)) { //each line is read in turn
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
	// this should be a shrinking, and should never fail.
	cLines = (char **)realloc(cLines, lLinesQty * sizeof (char*));
		mCheckMallocWorked(cLines)
	// 1 means ContActivation
	// 0 means discrete, and values of 2+ indicate parameter has been omitted,
	// defaulting to discrete, and value is actually # of samples in file
	rSes.rLearn->iEvalMode=atof(cLines[0])>1 ? 1 : 0; iArchLineIdx = 1 - rSes.rLearn->iEvalMode;
		if (rSes.rLearn->iEvalMode) printf ("Discrete output values indicated.\n");
	else printf("Continuous output values indicated.\n");

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
		rSes.lMemStructAlloc = rSes.lMemStructAlloc || RLEARNd; // flag existence of alllocation
	
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
		for (int k=1; k<=rSes.rLearn->iOutputQty; ++k){
			rSes.rLearn->dDOutputs[ IDX2C( k, s, rSes.rLearn->iOutputQty+1 ) ]=atof(pch); // convert and assign each value in a line
			pch = strtok (NULL, " ,\t");
		}
		rSes.rLearn->dXInputs[ IDX2C( 0, s, rSes.rLearn->iInputQty+1 ) ]=0.0; // virtual input zero should always be zero
		rSes.rLearn->dDOutputs[ IDX2C( 0, s, rSes.rLearn->iOutputQty+1) ]=0.0; // output neuron zero should always produce sector 0 output
		free(cLines[lCurrentLine]);
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
		}	
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ // now loop over output values
			if(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 )]>=rSes.rNet->iSectorQty){
				// if any output values fall beyond the maximum sector value, alert and make recommendation
				fprintf(stderr, "Error: Sample #%d has value that exceeds sector qty %d; suggest increasing to %d!\n",
					s, rSes.rNet->iSectorQty, static_cast<int>(floor(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ]*1.33)+1));
				++iOverK;
			}
		}
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
	rSes.rLearn->dSqrErr=(double*)malloc( OUTSIZED ); // array for RMSE calculation, changed to OUTSIZED 1/8/12
		mCheckMallocWorked(rSes.rLearn->dSqrErr)
	// allocate host complex arrays
	rSes.rLearn-> cdcXInputs  =(cuDoubleComplex*)malloc( INSIZECX ); // cx X Input signal
		mCheckMallocWorked(rSes.rLearn->cdcXInputs)
	rSes.rLearn-> cdcDOutputs =(cuDoubleComplex*)malloc( OUTSIZECX ); // cx D desired output signal
		mCheckMallocWorked(rSes.rLearn->cdcDOutputs)
	rSes.rLearn-> cdcYEval    =(cuDoubleComplex*)malloc( OUTSIZECX ); // cx evaluated Y output signal
		mCheckMallocWorked(rSes.rLearn->cdcYEval)
	rSes.rLearn-> cdcAltYEval =(cuDoubleComplex*)malloc( OUTSIZECX ); // cx alt evaluation Y output
		mCheckMallocWorked(rSes.rLearn->cdcAltYEval)
		rSes.lMemStructAlloc = rSes.lMemStructAlloc || RLEARNcdc; // flag existence of allocated structs

	for(long S=0;S<rSes.rLearn->lSampleQty; S++){
		for (int I=0;I< IQTY ; I++){
			rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )]
				= ConvScalarCx( rSes, rSes.rLearn-> dXInputs [IDX2C( I, S, IQTY )] ); // convert scalar inputs on host
		}
		for (int O=0;O< OQTY ; O++){
			rSes.rLearn-> dYEval      [IDX2C(  O, S, OQTY )] = S; 
			rSes.rLearn-> dAltYEval	  [IDX2C(  O, S, OQTY )] = -S;
			rSes.rLearn-> dSqrErr	  [IDX2C(  O, S, OQTY )] = O;
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
		cuRandomizeWeightsBlock(rSes); // populate network with random weight values
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
		//rSes.rNet->iNeuronOfst[L]=rSes.rNet->iNeuronOfst[L-1]+rSes.rNet->iNeuronQTY[L-1];
		//rSes.rNet->iWeightOfst[L]=rSes.rNet->iWeightOfst[L-1]+rSes.rNet->iWeightQTY[L-1];
		rSes.rNet->iNeuronOfst[L] = rSes.rNet->iNeuronOfst[L-1] + ((7+rSes.rNet->iNeuronQTY[L-1])/8)*8;
		rSes.rNet->iWeightOfst[L] = rSes.rNet->iWeightOfst[L-1] + ((7+rSes.rNet->iWeightQTY[L-1])/8)*8;
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
		rSes.rNet->dINV_S[L]=1.0/(double)DQTY; //printf("%d %d %f\n", L, DQTY, rSes.rNet->dINV_S[L]);
	}
	rSes.rNet->dINV_S[0]=1.0; // setup final fixed-length pointer

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
	}
	return iReturn;
}


long cBarge::LayersToBlocks(struct rohanContext& rSes) //, struct rohanNetwork& Net)
{mIDfunc /// moves weight values from old layer structures to new block structures
	// record fixed-length accumulation 
	struct rohanNetwork * rnSrc; //, * rnDest ;
	struct rohanLayer * rlSrc;
	long iReturn=0;
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
	
	//// blank original values
	//for (long i = 0; i<MAXWEIGHTS; ++i) rSes.rNet->Wt[i]=cdcZero;
	//for (long i = 0; i<MAXNEURONS; ++i) rSes.rNet->Deltas[i]=rSes.rNet->Signals[i]=cdcZero;
	//rSes.rNet->iNeuronQTY[0]=rSes.rLearn->iInputQty+1; // initialize with inputs for Layer Zero
	//rSes.rNet->iDendrtOfst[0]=rSes.rNet->iDendrtQTY[0]=rSes.rNet->iNeuronOfst[0]=rSes.rNet->iWeightOfst[0]=rSes.rNet->iWeightQTY[0]=0;

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
	}
	return iReturn;

}

long cBarge::LetWriteWeights(struct rohanContext& rSes)
{mIDfunc/// dump ASCII weight values to disk
	long lReturn;
	// dump weights for verification
	FILE *fileOutput; // File handle for output
	lReturn=AsciiFileHandleWrite(rSes.sRohanVerPath, "weightdump.txt", &fileOutput);
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
	lReturn=AsciiFileHandleWrite(rSes.sRohanVerPath, sFileAscii, &fileOutput);
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
	int iReturn=1;

	printf("cBarge diagnostics: ");
	if(Team==NULL)
		printf("No team!\n", iReturn=0);
	if(Drover==NULL)
		printf("No drover!\n", iReturn=0);
	if(rSes->rLearn->cdcXInputs==NULL)
		printf("No complex inputs at host!\n", iReturn=0);
	if(rSes->rLearn->dDOutputs==NULL)
		printf("No scalar outputs at host!\n", iReturn=0);
	if(rSes->rLearn->cdcDOutputs==NULL)
		printf("No complex outputs at host!\n", iReturn=0);
	if (rLearn==NULL)
		printf("No rLearn structure!\n", iReturn=0);
	else
		//printf("Holding %d samples w/ %d inputs, %d output(s)\n", *rLearn->lSampleQty, *rLearn->iInputQty, *rLearn->iOutputQty);
		printf("Barge is holding %d samples w/ %d inputs, %d output(s).\n", rSes->rLearn->lSampleQty, rSes->rLearn->iInputQty, rSes->rLearn->iOutputQty);
	
	return iReturn;
}


long cBarge::DoCuFree(struct rohanContext &rSes)
{mIDfunc/// free allocated memory for all structures
	
	cuFreeNNTop(rSes); // free network topology structures
	cuFreeLearnSet(rSes); // free learning set structures
	
	return 0;
}


long cBarge::cuFreeNNTop(struct rohanContext &rSes)
{mIDfunc/// frees data structures related to network topology
	int iFreed=0;
	if (rSes.lMemStructAlloc && RNETbdry) {//check
		free( rSes.rNet->cdcSectorBdry );
		rSes.lMemStructAlloc = rSes.lMemStructAlloc && !RNETbdry; 
		++iFreed;
	}
	// layer components
	if (rSes.lMemStructAlloc && RNETlayers){
		free( rSes.rNet->rLayer[0].ZOutputs ); // Layer Zero has no need of weights!
		for (int i=1; i < rSes.rNet->iLayerQty; ++i){ 
			struct rohanLayer& lay=rSes.rNet->rLayer[i];
		
			free( lay.Weights ); // free the weights
			free( lay.Deltas ); // free the backprop areas
			free( lay.XInputs ); // free the inputs
			free( lay.ZOutputs ); // free the outputs
		}
		free( rSes.rNet->rLayer ); // free empty layers
		rSes.lMemStructAlloc = rSes.lMemStructAlloc && !RNETlayers; 
		++iFreed;
	}	
	if(iFreed)
		printf("Network structures freed.\n");
	return iFreed;
}


long cBarge::cuFreeLearnSet(struct rohanContext &rSes)
{mIDfunc/// free the learning set of samples
	int iFreed=0;
	if (rSes.lMemStructAlloc && RLEARNd) { //check
		free( rSes.rLearn->dXInputs ); 
		free( rSes.rLearn->dDOutputs );
		free( rSes.rLearn->dYEval ); 
		free( rSes.rLearn->dAltYEval ); 
		free( rSes.rLearn->dSqrErr );
		rSes.lMemStructAlloc = rSes.lMemStructAlloc && !RLEARNd; 
		++iFreed;
	}
	
	if (rSes.lMemStructAlloc && RLEARNcdc){ // check
		free( rSes.rLearn->cdcXInputs ); 
		free( rSes.rLearn->cdcDOutputs ); 
		free( rSes.rLearn->cdcYEval ); 
		free( rSes.rLearn->cdcAltYEval );
		rSes.lMemStructAlloc = rSes.lMemStructAlloc && !RLEARNcdc; 
		++iFreed;
	}
	if(iFreed)
		printf("Learning set structures freed.\n");
	return iFreed;
}
