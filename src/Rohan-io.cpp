/* Includes, cuda */
#include "cuda.h"

#include <iostream>
using namespace std;
#include "Rohan.h"
#include "Rohan-data.h"
#include "Rohan-learn.h"
#include "ShowMe.h"
#include <conio.h> //for _getch
#define TWO_PI 6.283185307179586476925286766558
//#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))

extern int iDebugLvl, iWarnings, iErrors, iTrace;
extern long bCUDAavailable;

int cuMessage(cublasStatus csStatus, char *sName, char *sCodeFile, int iLine, char *sFunc)
{	
	char *sMsg;

	switch (csStatus) {
		case CUBLAS_STATUS_SUCCESS: sMsg=_strdup("operation completed successfully");
			break;
		case CUBLAS_STATUS_NOT_INITIALIZED: sMsg=_strdup("library not initialized");
			break;
		case CUBLAS_STATUS_ALLOC_FAILED: sMsg=_strdup("resource allocation failed");
			break;
		case CUBLAS_STATUS_INVALID_VALUE: sMsg=_strdup("unsupported numerical value was passed to function");
			break;
		case CUBLAS_STATUS_ARCH_MISMATCH: sMsg=_strdup("function requires an architectural feature absent from the architecture of the device");
			break;
		case CUBLAS_STATUS_MAPPING_ERROR: sMsg=_strdup("access to GPU memory space failed");
			break;
		case CUBLAS_STATUS_EXECUTION_FAILED: sMsg=_strdup("GPU program failed to execute");
			break;
		case CUBLAS_STATUS_INTERNAL_ERROR: sMsg=_strdup("an internal operation failed");
			break;
		default: sMsg=_strdup("unknown response");
	}
	fprintf(stderr,"%s %s line %i: CUBLAS %s: %s\n", sCodeFile, sFunc, iLine, sMsg, sName);
	return 0;
}


int BinaryFileHandleRead(char* sFileName, FILE** fileInput)
{mIDfunc/// Opens a file for reading in binary mode, typically a .wgt weight file.
	*fileInput = fopen(sFileName, "rb");  /* Open in BINARY mode */
	if (*fileInput == NULL) {
		fprintf(stderr, "Error opening %s for reading.\n", sFileName);
		return 0;
	}
	else return 1;
}

long BinaryFileHandleWrite(char *sFileName, FILE **fileOutput)
{mIDfunc/// Opens a file for writing in binary mode, typically top record results of a learning sewssion and/or to save human-readable weight values.
	*fileOutput = fopen(sFileName, "wb");  /* Open in BINARY mode */
	if (*fileOutput == NULL) {
		fprintf(stderr, "Error opening %s for writing.\n", sFileName);
		return 0;
	}
	else return 1;
}


int AsciiFileHandleRead(char *sFileName, FILE **fileInput)
{mIDfunc/// Opens a file for reading in ASCII mode, typically the .txt learning set file.
	*fileInput = fopen(sFileName, "r");  /* Open in ASCII mode */
	if (*fileInput == NULL) {
		fprintf(stderr, "Error opening %s for reading.\n", sFileName);
		return 0;
	}
	else return 1;
}

long AsciiFileHandleWrite(char *sFileName, FILE **fileOutput)
{mIDfunc/// Opens a file for writing in ASCII mode, typically top record results of a learning sewssion and/or to save human-readable weight values.
	*fileOutput = fopen(sFileName, "w");  /* Open in ASCII mode */
	if (*fileOutput == NULL) {
		fprintf(stderr, "Error opening %s for writing.\n", sFileName);
		return 0;
	}
	else return 1;
}

int cuMakeLayers(int iInputQty, char *sLayerSizes, struct rohanContext& rSes)
{mIDfunc
/// Parses a string to assign network architecture parameters for use by later functions. 
/// Returns neurons in last layer if successful, otherwise 0
	char *sArchDup, *sDummy;
	int iLayerQty=1;

	sArchDup = _strdup(sLayerSizes); // strtok chops up the input string, so we must make a copy (or do we? - 6/15/10) (yes we do 8/22/10)
	sDummy = strtok (sArchDup, " ,\t");
	while (sDummy!=NULL) {// this loop counts the values present in a copy of sLayerSizes, representing neurons in each layer until a not-legal layer value is reached
		sDummy = strtok (NULL, " ,\t");
		++iLayerQty; //count layers
		//printf("%d-%s\n",iLayerQty, sDummy);
	}
	rSes.rNet->rLayer=(struct rohanLayer*)malloc(iLayerQty * sizeof (struct rohanLayer)); //point to array of layers
		mCheckMallocWorked(rSes.rNet->rLayer)
	printf("%d layers plus input layer allocated.\n", (iLayerQty-1));
	
	sArchDup=_strdup(sLayerSizes); // second pass
	sDummy = strtok(sArchDup, " ,\t");
	for (int i=0;i<iLayerQty;++i) {// this loop stores neurons in each layer
		//printf("%d-%s/%d\n",i, sDummy, atoi(sDummy));
		//printf ("%s Layer %d neurons %d\n",sDummy, i, rSes.rNet->rLayer[i].iNeuronQty);
		if (i) {
			rSes.rNet->rLayer[i].iNeuronQty = atoi(sDummy);
			rSes.rNet->rLayer[i].iDendriteQty=rSes.rNet->rLayer[i-1].iNeuronQty; //previous layer's neuron qty is dendrite qty
			sDummy = strtok (NULL, " ,\t");
		}
		else {
			rSes.rNet->rLayer[i].iNeuronQty = iInputQty; // layer zero has virtual neurons with outputs equal to inputs converted to phases
			rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		}
		printf ("Layer %d: %d nodes\n", i, rSes.rNet->rLayer[i].iNeuronQty);
	}
	if (cuMakeNNStructures(rSes)) 
		printf("Nodes allocated.");
	return rSes.rNet->rLayer[rSes.rNet->iLayerQty-1].iNeuronQty;
}



int cuMakeArchValues(char *sMLMVNarch, struct rohanContext& rSes)
{mIDfunc/// Parses a string to assign network architecture parameters for use by later functions. 
/// Returns neurons in last layer if successful, otherwise 0
	char *sArchDup, *sDummy;
	sArchDup = _strdup(sMLMVNarch); // strtok chops up the input string, so we must make a copy (or do we? - 6/15/10) (yes we do 8/22/10)
	sDummy = strtok(sArchDup, " ,\t"); // first value is always # of samples, to be skipped
	sDummy = strtok(NULL, " ,\t"); // second value is always # of sectors
	rSes.rNet->iSectorQty = atoi(sDummy); 
	rSes.rNet->kdiv2= atoi(sDummy)/2;
	rSes.rNet->dK_DIV_TWO_PI = rSes.rNet->iSectorQty / TWO_PI; // calc this now to prevents redundant conversion operations
	rSes.rNet->iLayerQty = 0;
	sDummy = strtok (NULL, " ,\t");
	while (atoi(sDummy)) {// this loop counts the values present in a copy of cMLMVNarch, representing neurons in each layer until a not-legal layer value is reached
		//printf ("%s\n",cDummy);
		sDummy = strtok (NULL, " ,\t");
		++rSes.rNet->iLayerQty; //count layers
		//printf("%s %d layers \"%s", sDummy, rSes.rNet->iLayerQty, sMLMVNarch);
	}
	
	rSes.rNet->rLayer=(struct rohanLayer*)malloc(rSes.rNet->iLayerQty * sizeof (struct rohanLayer)); //point to array of layers
		mCheckMallocWorked(rSes.rNet->rLayer)
	
	if (sDummy!=NULL) {// check that there is another parameter, not just the end of the string
		rSes.rNet->sWeightSet=_strdup(sDummy);
		printf("Using weights in %s\n", rSes.rNet->sWeightSet);}
	else{
		printf("No weight set filename specificed, get from config file or cli args.\n");
		rSes.rNet->sWeightSet="NOTFOUND";
	}
	sDummy = strtok (NULL, " ,\t");
	if (sDummy!=NULL) {// check that there is another parameter, not just the end of the string
		rSes.rNet->iWeightMode=atoi(sDummy);
		printf("Weight mode %d specified\n", rSes.rNet->iWeightMode); }
	else{
		printf("No weight mode specificed, get from config file or cli args.\n");
		rSes.rNet->iWeightMode=0;
	}
	
	sArchDup=_strdup(sMLMVNarch); // second pass
	sDummy = strtok(sArchDup, " ,\t"); // skip sample qty
	sDummy = strtok(NULL, " ,\t"); // skip sector qty
	//rSes.rNet->iLayerQty = 0;
	int l=0;
	sDummy = strtok (NULL, " ,\t");
	while (atoi(sDummy)) {// this loop stores neurons in each layer, until it encounrs an invalid neuron qty
		//printf ("%s %d\n",cDummy, iLayerQty);
		//(*iNeuronQty)[*iLayerQty]=atoi(cDummy);
		rSes.rNet->rLayer[l].iNeuronQty = atoi(sDummy);
		if (l) rSes.rNet->rLayer[l].iDendriteQty=rSes.rNet->rLayer[l-1].iNeuronQty; //previous layer's neuron qty is dendrite qty
		else rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		sDummy = strtok (NULL, " ,\t");
		++l; //count layers
	}
	/*if(rSes.bCUDAavailable)
		devCopyArchValues(rSes);*/
	//for(int i=0;i<*iLayerQty;++i) printf("Layer %d Neurons %d\n", i, (*iNeuronQty)[i]);
	mDebug(1,0) for (int i=0; i<rSes.rNet->iLayerQty; ++i) printf("%s line %d: layer %d neurons %d dendrites %d\n", __FILE__, __LINE__, i, rSes.rNet->rLayer[i].iNeuronQty, rSes.rNet->rLayer[i].iDendriteQty);
	//return (*iNeuronQty)[(*iLayerQty)-1];
	mDebug(1,0) printf("cuMakeArchValues returns.\n");
	cout << "NN architecture made" << endl;
	return rSes.rNet->rLayer[rSes.rNet->iLayerQty-1].iNeuronQty;
}


long cuReLoadSampleSet(struct rohanContext& rSes, FILE *fileInput)
{mIDfunc	/// pulls in values from .txt files, used during main loop after NN arch is already made
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
	// 0 means Continuous Activation, 
	//		0
	//		150 10 4 4 1 iris.wgt 0
	//		5.1, 3.5, 1.4, 0.2, 2.5, Iris-setosa
	//		7.0, 3.2, 4.7, 1.4, 5.0, Iris-versicolor
	// 1 means discrete, for integral problems or identification/categorization
	//		1
	//		416 13 200 4 1 PC-63-32-200.wgt 0
	//		5.0444	1.7085	3.1587	3.5219	3.0372	3.3963
	// values of 2+ indicate parameter has been omitted, defaulting to discrete, and value is actually # of samples in file
	//		10000 384 9 36 1 AirplanePsDN1W3S10kRMSE876.wgt 0
	//		201  225  194  203  230  221  195  229  186      3
	//		215  213  222  206  210  218  215  227  208      3
	rSes.rLearn->iEvalMode=atoi(cLines[0]); // let eval mode be read from beginning of first line
	rSes.rLearn->iEvalMode=atoi(cLines[0])>1 ? 1 : rSes.rLearn->iEvalMode; // if it is 2 or more, default to 0-discrete, otherwise remain as read
	// if first line starts with value greater than 1, it is the line for sample qty and net arch
	iArchLineIdx = atoi(cLines[0])>1 ? 0 : 1; // otherwise the second line is for that
	if (rSes.rLearn->iEvalMode) 
		printf ("Discrete output values indicated.\n");
	else 
		printf("Continuous output values indicated.\n");
		mDebug(1, 0) printf("\"%s\"%siEvalMode %d, iArchLineIdx %d\n", cLines[0], cLines[1], rSes.rLearn->iEvalMode, iArchLineIdx);

	char *sArch, *tok;
	//int iTokQty;
	sArch = _strdup(cLines[iArchLineIdx]); // make a sacrificial copy
	tok = strtok(sArch, " ,\t");
	if (sArch==NULL) // no additional params present
		printf("No params present; fetching from topology.\n");
	else 
		printf("Params present but ignored; fetching from existing topology.\n");
		//cuMakeArchValues(cLines[iArchLineIdx], rSes);
	
	rSes.rLearn->iInputQty=rSes.rNet->rLayer[0].iNeuronQty;
	rSes.rLearn->iOutputQty=rSes.rNet->rLayer[rSes.rNet->iLayerQty-1].iNeuronQty;
	
		mDebug(1, 0) printf("\"%s%s %d outputs", cLines[iArchLineIdx], rSes.rNet->sWeightSet, rSes.rLearn->iOutputQty);

	
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
		mDebug(1,0) printf("%d inputs, %d outputs, %d values per line.\n",rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty, rSes.rLearn->iValuesPerLine);
	printf("%d inputs, %d output(s) each.\n", rSes.rLearn->iValuesPerLine-rSes.rLearn->iOutputQty, rSes.rLearn->iOutputQty);
	int iExcessValueQty=rSes.rLearn->iValuesPerLine-rSes.rLearn->iInputQty-rSes.rLearn->iOutputQty;
	if(iExcessValueQty>0) {
		fprintf(stderr, "Warning: %d unused values in sample tuples.\n", iExcessValueQty);
		++rSes.iWarnings;
	}
	if(iExcessValueQty<0) {
		fprintf(stderr, "Error: %d values not found in sample tuples.\n", iExcessValueQty*-1);
		++rSes.iErrors;
	}
	
	//rSes.rLearn->rSample = (rohanSample*)malloc(rSes.rLearn->lSampleQty * sizeof (rohanSample)); //allocate one pointer to a sample structure per line
	//	mCheckMallocWorked(rSes.rLearn->rSample)
	rSes.rLearn->dXInputs=(double*)malloc((rSes.rLearn->iInputQty+1) * rSes.rLearn->lSampleQty * sizeof(double)); // allocate a row of so-many double pointers
			mCheckMallocWorked(rSes.rLearn->dXInputs)
	rSes.rLearn->dYEval=(double*)malloc((rSes.rLearn->iOutputQty+1) * rSes.rLearn->lSampleQty * sizeof(double)); // allocate a row of so-many double pointers
			mCheckMallocWorked(rSes.rLearn->dYEval)
		

	for (long s=0; s<rSes.rLearn->lSampleQty; ++s){ //iterate over the number of samples
		//struct rohanSample& sam = rSes.rLearn->rSample[s]; 
		//sam.dXInputs=(double*)malloc((rSes.rLearn->iValuesPerLine+1) * sizeof(double)); // allocate a row of so-many double pointers
		//	mCheckMallocWorked(sam.dXInputs)
		//sam.dYEval=(double*)malloc((rSes.rLearn->iOutputQty+1) * sizeof(double)); // allocate a row of so-many double pointers
		//	mCheckMallocWorked(sam.dYEval)
		pch = strtok (cLines[lCurrentLine], " ,\t"); // get the first token on a line
		if(rSes.bRInJMode){ // if flag for compatibility with older NN simulator is set
			for (int k=rSes.rLearn->iInputQty; k>=1; --k){ // save it beginning with last position
				rSes.rLearn->dXInputs[IDX2C( k, s, (rSes.rLearn->iInputQty+1) )]=atof(pch); // convert and assign each value in a line
				pch = strtok (NULL, " ,\t");
			}
		}
		else{ // otherwise store things the usual way
			for (int k=1; k<=rSes.rLearn->iInputQty; ++k){ // save it beginning with position 1
				//sam.dXInputs[k]=atof(pch); // convert and assign each value in a line
				rSes.rLearn->dXInputs[IDX2C( k, s, (rSes.rLearn->iInputQty+1) )]=atof(pch); // convert and assign each value in a line
				pch = strtok (NULL, " ,\t");
			}
		}
		rSes.rLearn->dXInputs[IDX2C( 0, s, (rSes.rLearn->iInputQty+1) )]=-999.0; // mark unused slot zero
		for (int k=0; k<=rSes.rLearn->iOutputQty; ++k){ // mark all eval outputs
			rSes.rLearn->dYEval[IDX2C( k, s, (rSes.rLearn->iOutputQty+1) )]=-999.0;
		}
		free(cLines[lCurrentLine]); //if (lCurrentLine<10) printf("freed cLine[%d]\n", lCurrentLine);
		++lCurrentLine;
	}
	
	free(cLines[0]);
	if (iArchLineIdx) free(cLines[1]); // WEIRD MEMORY ERRORS? LOOK HERE
	// above line avoids double-freeing cLines[1] if it was used for a sample instead of the sample qty
	free(cLines);
 	return lLinesQty; // returns qty of lines read from file, not the same as quantity of samples
}


long cuNNLoadWeights(struct rohanContext &rSes, FILE *fileInput)
{mIDfunc
// pulls in values from .wgt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	long lReturnValue=0;
	for (int j=1; j < rSes.rNet->iLayerQty; ++j){ //no weights for layer 0
		struct rohanLayer& lay = rSes.rNet->rLayer[j];
		for (int k=1; k <= lay.iNeuronQty; ++k){ // no weights for neuron 0
			for (int i=0; i <= lay.iDendriteQty; ++i){
				cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
				fread(&(way.x), sizeof(double), 1, fileInput);
				fread(&(way.y), sizeof(double), 1, fileInput);
				//printf("L%d N%d D%d %f+%f\n", j, k, i, way.x, way.y);
				++lReturnValue;
			}
		}
	}
	fclose(fileInput);
		//ShowMeWS(rSes, false);
	return lReturnValue;
}

long cuSaveNNWeights(struct rohanContext &rSes, FILE *fileOutput)
{mIDfunc
// writes values to .wgt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	long lReturnValue=0;
	//cublasStatus csStatus;
	//for (int j=1; j < rSes.rNet->iLayerQty; ++j){ //no weights for layer 0
	struct rohanNetwork& Net = *rSes.rNet;
	for (int LAY=1; LAY<Net.iLayerQty; ++LAY){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
			//for (int i=0; i <= lay.iDendriteQty; ++i){
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				//cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				fwrite(&(way.x), sizeof(double), 1, fileOutput);
				fwrite(&(way.y), sizeof(double), 1, fileOutput);
				++lReturnValue;
				//mDebug(4,0) printf("weight %d.%d.%d= % 1f + % 1f i\n",  i, j, k, way.x, way.y);
			}
		}
	}
	fclose(fileOutput);
		//ShowMeWS(rSes.rNet, false);
	return lReturnValue;
}

long cuSaveNNWeightsASCII(struct rohanContext &rSes, FILE *fileOutput)
{mIDfunc
// writes values to .txt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	long lReturnValue=0;
	//cublasStatus csStatus;
	//for (int j=1; j < rSes.rNet->iLayerQty; ++j){ //no weights for layer 0
	struct rohanNetwork& Net = *rSes.rNet;
	for (int LAY=1; LAY<Net.iLayerQty; ++LAY){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
			//for (int i=0; i <= lay.iDendriteQty; ++i){
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				//cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				fprintf(fileOutput, "% 11f,% 11f,% d,% d,% d\n", way.x, way.y, LAY, k, i);
				++lReturnValue;
				//mDebug(4,0) printf("weight %d.%d.%d= % 1f + % 1f i\n",  j, k, i, way.x, way.y);
			}
		}
	}
	fclose(fileOutput);
		//ShowMeWS(rSes.rNet, false);
	return lReturnValue;
}

long cuPreSaveNNWeights(struct rohanContext& rSes)
{mIDfunc
	FILE *fileOutput;
	char sFileName[255]; //="DefaultSession";
	char sFileAscii[255]; //="DefaultSession";

	strncpy(sFileName,rSes.sSesName,250); // do not exceed 254 char file name
	strcat(sFileName,".wgt");
	strncpy(sFileAscii,rSes.sSesName,248); // do not exceed 254 char file name
	strcat(sFileAscii,"WGT.txt");

	fileOutput = fopen(sFileName, "wb");  /* Open in BINARY mode */
	if (fileOutput == NULL) {
		fprintf(stderr, "Error opening %s for writing.\n", sFileName);
		++rSes.iErrors;
		return 0;
	}
	else {
		long lWWrit=cuSaveNNWeights(rSes, fileOutput);
		printf("%d binary weights written to %s\n", lWWrit, sFileName);
		fileOutput = fopen(sFileAscii, "w");  /* Open in ASCII mode */
		if (fileOutput == NULL) {
			fprintf(stderr, "Error opening %s for writing.\n", sFileAscii);
			++rSes.iErrors;
			return 0;
		}
		else {
			long lWWrit=cuSaveNNWeightsASCII(rSes, fileOutput);
			printf("%d ASCII weights written to %s\n", lWWrit, sFileAscii);
			return 1;
		}	
	}
}

long AsciiWeightDump(struct rohanContext& rSes, FILE *fileOutput)
{mIDfunc
/// outputs values from .wgt files as ASCII text
/// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	long lReturnValue=0;

	fprintf(fileOutput, "REAL\tIMAGINARY\tLAYER\tNEURON\tINPUT\n");
	struct rohanNetwork& Net = *rSes.rNet;
	//for (int j=1; j < rSes.rNet->iLayerQty; ++j){
	for (int LAY=1; LAY<Net.iLayerQty; ++LAY){
		//struct rohanLayer& lay = rSes.rNet->rLayer[j];
		//for (int k=1; k <= lay.iNeuronQty; ++k){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
			//for (int i=0; i <= lay.iDendriteQty; ++i){
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				//cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				++lReturnValue;
				//fprintf(fileOutput, "%f\t%f\t%d\t%d\t%d\n", lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)].x, lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)].y, j, k, i);
				fprintf(fileOutput, "%f\t%f\t%d\t%d\t%d\n", way.x, way.y, LAY, k, i);
				
			}
		}
	}
	fclose(fileOutput);
	return lReturnValue;
}

