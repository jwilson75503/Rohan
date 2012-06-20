/* Includes, cuda */
#include "stdafx.h"

extern int gDebugLvl, gTrace;
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
	*fileInput = fopen(sFileName, "r");  /* Open in ASCII read mode */
	if (*fileInput == NULL) {
		fprintf(stderr, "Error opening %s for reading.\n", sFileName);
		return 0;
	}
	else return 1;
}

int AsciiFileHandleWrite(char *sFilePath, char *sFileName, FILE **fileOutput)
{mIDfunc/// Opens a file for writing in ASCII mode, typically to record results of a learning session and/or to save human-readable weight values.
	char sString[MAX_PATH];

	if(DirectoryEnsure(sFilePath)){
		sprintf(sString, "%s\\%s", sFilePath, sFileName);
		*fileOutput = fopen(sString, "w");  /* Open in ASCII write mode */
		if (*fileOutput == NULL) {
			errPrintf("Error opening %s for writing.\n", sString);
			return false;
		}
		else return true;
	}
	else{
		errPrintf("Error making %s for writing.\n", sFilePath);
		return false;
	}
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
	}
	rSes.rNet->rLayer=(struct rohanLayer*)malloc(iLayerQty * sizeof (struct rohanLayer)); //point to array of layers
		mCheckMallocWorked(rSes.rNet->rLayer)
		rSes.lMemStructAlloc = rSes.lMemStructAlloc || RNETlayers;
	printf("%d layers plus input layer allocated.\n", (iLayerQty-1));
	
	sArchDup=_strdup(sLayerSizes); // second pass
	sDummy = strtok(sArchDup, " ,\t");
	for (int i=0;i<iLayerQty;++i) {// this loop stores neurons in each layer
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
		sDummy = strtok (NULL, " ,\t");
		++rSes.rNet->iLayerQty; //count layers
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
		rSes.rNet->bContActivation=atoi(sDummy);
		if(rSes.rNet->bContActivation)
			printf("Continuous activation mode specified\n"); 
		else
			printf("Discrete activation mode specified\n"); 
	}
	else{
		printf("No Activation mode specificed, get from config file or cli args.\n");
	}
	
	sArchDup=_strdup(sMLMVNarch); // second pass
	sDummy = strtok(sArchDup, " ,\t"); // skip sample qty
	sDummy = strtok(NULL, " ,\t"); // skip sector qty
	int l=0;
	sDummy = strtok (NULL, " ,\t");
	while (atoi(sDummy)) {// this loop stores neurons in each layer, until it encounters an invalid neuron qty
		rSes.rNet->rLayer[l].iNeuronQty = atoi(sDummy);
		if (l) rSes.rNet->rLayer[l].iDendriteQty=rSes.rNet->rLayer[l-1].iNeuronQty; //previous layer's neuron qty is dendrite qty
		else rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		sDummy = strtok (NULL, " ,\t");
		++l; //count layers
	}
	mDebug(1,0) for (int i=0; i<rSes.rNet->iLayerQty; ++i) printf("%s line %d: layer %d neurons %d dendrites %d\n", __FILE__, __LINE__, i, rSes.rNet->rLayer[i].iNeuronQty, rSes.rNet->rLayer[i].iDendriteQty);
	mDebug(1,0) printf("cuMakeArchValues returns.\n");
	cout << "NN architecture made" << endl;
	
	return rSes.rNet->rLayer[rSes.rNet->iLayerQty-1].iNeuronQty;
}


long cuNNLoadWeights(struct rohanContext &rSes, FILE *fileInput)
{mIDfunc
// pulls in values from .wgt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	long lReturnValue=0, lElementsReturned=0;
	for (int j=1; j < rSes.rNet->iLayerQty; ++j){ //no weights for layer 0
		struct rohanLayer& lay = rSes.rNet->rLayer[j];
		for (int k=1; k <= lay.iNeuronQty; ++k){ // no weights for neuron 0
			for (int i=0; i <= lay.iDendriteQty; ++i){
				cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
				lElementsReturned+=fread(&(way.x), sizeof(double), 1, fileInput); // count elements as we read weright values
				lElementsReturned+=fread(&(way.y), sizeof(double), 1, fileInput);
				++lReturnValue;
			}
		}
	}
	fclose(fileInput);
	
	if(lElementsReturned != (lReturnValue*2) ){ // not enough data, raise an alarm
		errPrintf("WARNING! Read past end of weight data. ", ++rSes.iWarnings);
		errPrintf("Found %d doubles, needed ", lElementsReturned);
		errPrintf("%d (2 per complex weight).\n", lReturnValue*2);
	}

	return lReturnValue;
}

long cuSaveNNWeights(struct rohanContext &rSes, FILE *fileOutput)
{mIDfunc
// writes values to .wgt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	long lReturnValue=0;

	struct rohanNetwork& Net = *rSes.rNet;
	for (int LAY=1; LAY<Net.iLayerQty; ++LAY){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				fwrite(&(way.x), sizeof(double), 1, fileOutput);
				fwrite(&(way.y), sizeof(double), 1, fileOutput);
				++lReturnValue;
			}
		}
	}
	fclose(fileOutput);

	return lReturnValue;
}

long cuSaveNNWeightsASCII(struct rohanContext &rSes, FILE *fileOutput)
{mIDfunc
// writes values to .txt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	long lReturnValue=0;

	struct rohanNetwork& Net = *rSes.rNet;
	for (int LAY=1; LAY<Net.iLayerQty; ++LAY){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				fprintf(fileOutput, "% 11f,% 11f,% d,% d,% d\n", way.x, way.y, LAY, k, i);
				++lReturnValue;
			}
		}
	}
	fclose(fileOutput);
	
	return lReturnValue;
}

long cuPreSaveNNWeights(struct rohanContext& rSes)
{mIDfunc
	FILE *fileOutput;
	char sFileName[255]; //="DefaultSession";
	char sFileAscii[255]; //="DefaultSession";

	strncpy(sFileName,rSes.sSesName,250); // do not exceed 254 char file name
	strcat(sFileName,"Rmse");
	sprintf(sFileName,"%s%d", sFileName, (int)(rSes.dDevRMSE*100));
	strncpy(sFileAscii,sFileName,248); // do not exceed 254 char file name
	strcat(sFileName,".wgt");
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
	
	for (int LAY=1; LAY<Net.iLayerQty; ++LAY){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				++lReturnValue;
				fprintf(fileOutput, "%f\t%f\t%d\t%d\t%d\n", way.x, way.y, LAY, k, i);
			}
		}
	}
	fclose(fileOutput);
	return lReturnValue;
}

