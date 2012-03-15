#include "stdafx.h"

/* Boost utilities */
#include <boost/timer/timer.hpp>

//#include <time.h> // for tsrtuct
#include <sys/timeb.h>
using namespace std;
using std::cin;
using std::cout;

#define TWO_PI 6.283185307179586476925286766558
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))

extern int iDebugLvl, iWarnings, iErrors, iTrace;
extern long bCUDAavailable;

//////////////// class cDrover begins ////////////////

void cDrover::ShowMe()
{
	//ShowMeSes(* rSes, false);
	printf("Am Volga boatman.\n");
}


long cDrover::SetContext( rohanContext& rC)
{/// enables pointer access to master context struct
	rSes = &rC;
	return 0;
}


long cDrover::SetDroverBargeAndTeam( class cDrover * cdDrover, class cBarge * cbBarge, class cDeviceTeam * cdtTeam)
{mIDfunc /// sets pointers to hitch barge to team and mount driver on barge
	Barge = cbBarge;
	Team = cdtTeam;
	Barge->SetDrover(cdDrover);
	Barge->SetTeam(Team);
	Team->SetDrover(cdDrover);
	Team->SetBarge(Barge);
	return 0;
}


long cDrover::DoAnteLoop(int argc, char * argv[], class cDrover * cdDrover, class cBarge * cbBarge, class cDeviceTeam * cdtTeam)
{mIDfunc /// This function prepares all parameters, contexts, and data structures necesary for learning and evaluation.
	int iReturn=0;

	// initiate relationship
	SetDroverBargeAndTeam( cdDrover, cbBarge, cdtTeam);
	
	strcpy(rSes->sCLargsOpt, "blank");
	strcpy(rSes->sConfigFileOpt,"blank");
	strcpy(rSes->sConsoleOpt,"blank");
	strcpy(rSes->sLearnSetSpecOpt,"blank");
		
	if (argc) {
		rSes->bConsoleUsed=false;
		rSes->bCLargsUsed=true;
		//strcpy(rSes->sCLargsOpt, argv[0]); // point to first argument
		// call function to parse CLI args here
	}
	else {
		rSes->bConsoleUsed=true;
		rSes->bCLargsUsed=false;
	} /// potentially abusable 
	printf("Rohan v%s Neural Network Application\n", VERSION);
	iTrace=0;
	if (cdtTeam->CUDAverify(*rSes)){
		cout << "CUDA present" << endl;
		iReturn=ObtainGlobalSettings(*rSes);
		iReturn=ShowDiagnostics(*rSes);
		Barge->ShowDiagnostics();
		}
	else {
		fprintf(stderr, "Unrecoverable Error: No CUDA hardware or no CUDA functions present.\n", ++rSes->iErrors);
		iReturn=0;
	}
	return iReturn;
}


long cDrover::ObtainGlobalSettings(struct rohanContext& rSes)
{mIDfunc /// sets initial and default value for globals and settings
	int iReturn=1;
	
	//	globals
	iTrace=0; 
	iDebugLvl=0; 
	// session accrual
	rSes.iWarnings=0; rSes.iErrors=0; 
	rSes.lSampleQtyReq=0;
	// session modes
	rSes.bContActivation=false; 
	rSes.bRInJMode=false; 
	rSes.bRMSEon=true; 
	rSes.iEpochLength=1000; 
	rSes.iEvalBlocks=32; 
	rSes.iEvalThreads=16; 
	rSes.iBpropBlocks=1; 
	rSes.iBpropThreads=32; 
	rSes.iSaveInputs=0 /* include inputs when saving evaluations */;
	rSes.iSaveOutputs=0 /* include desired outputs when saving evaluations */;
	rSes.iSaveSampleIndex=0 /* includes sample serials when saving evaluations */;

	strcpy(rSes.sSesName,"DefaultSession");
	rSes.rLearn->bContInputs=true; 
	rSes.rLearn->iContOutputs=false; 

	if (iReturn){
		if(iTrace) cout << "Tracing is ON.\n" ;
		if (iDebugLvl) cout << "Debug level is " << iDebugLvl << "\n" ;
		cout << "Session warning and session error counts reset.\n";
		rSes.bContActivation ? cout << "Activation default is CONTINUOUS.\n" : cout << "Activation default is DISCRETE.\n"; // XX defaulting to false makes all kinds of heck on the GPU
		rSes.bRInJMode ? cout << "Reversed Input Order is ON.\n" : cout << "Reversed Input Order is OFF.\n"; // this is working backward for some reason 2/08/11 // still fubared 3/7/12 XX
		cout << "RMSE stop condition is ON. XX\n"; //
		cout << "Epoch length is 1000 iterations.\n";
		cout << rSes.iEvalBlocks << " EVAL Blocks per Kernel, " << rSes.iEvalThreads << " EVAL Threads per Block.\n";
		cout << rSes.iBpropBlocks << " BPROP Blocks per Kernel, " << rSes.iBpropThreads << " BPROP Threads per Block.\n";
		cout << "Continuous Inputs true by DEFAULT.\n";
		cout << "Continuous Outputs true by DEFAULT.\n";
	}

	return iReturn;
}


long cDrover::ShowDiagnostics(struct rohanContext& rSes)
{mIDfunc /// show some statistics, dump weights, and display warning and error counts
	double devdRMSE=0.0;
	int iReturn=1;
	long lSamplesEvald;
	cuDoubleComplex MinWts[MAXWEIGHTS]; // 16 x 2048
	
	AskSampleSetName(rSes);
	Barge->ObtainSampleSet(rSes);
	iReturn=Barge->DoPrepareNetwork(rSes);	
	Team->LetHitch(rSes);
	printf("\n\nBEGIN RMSE/EVALUATE TEST\n\n");
	printf("Team.RMSE = %f\n", Team->GetRmseNN(rSes, 0));
	Team->LetSlack(rSes);
	
	//printf("cDrover diagnostics: ");
	lSamplesEvald = cuEvalNNLearnSet(rSes);
		if (lSamplesEvald) printf("%d samples evaluated, ", lSamplesEvald);
		else {printf("No samples evaluated by cuEvalNNLearnSet\n");
			++rSes.iErrors;
			printf("Waiting on keystroke...\n"); _getch(); return iReturn;
		}

	int iDifferent = OutputValidate(rSes);
	printf("%d differences found on verify.\n", iDifferent);
	rSes.dRMSE = RmseNN(rSes, 0);
	printf("cuLearn.RMSE = %f\n", rSes.dRMSE);
	if(iDifferent) printf("\nEVALUATE FAIL\n\n");
	else printf("\nEVALUATE PASS\n\n");

	// some illustrative default values
	rSes.dTargetRMSE=floor(rSes.dRMSE-1.0)+1.0;
	rSes.dMAX=floor((double)abs(rSes.dRMSE-1.0));
	//printf("RMSE target is %e.\n", rSes.dTargetRMSE);
	//printf("MAX threshold is %e.\n", rSes.dMAX);
	
	printf("\n\nBEGIN TRAINABLE TEST\n\n");
	// begin trainable sample test DEVICE
	int iTrainable = Team->LetTrainNNThresh(rSes, rSes.rLearn->lSampleQty, 1, 'E', rSes.dTargetRMSE, rSes.iEpochLength);
	printf("%d trainable sample outputs found on device.\n", iTrainable);
	// begin trainable sample test HOST
	printf("HOST: Time to complete trainable determination:\n");
	int iHostTrainable;
	{
		boost::timer::auto_cpu_timer t;
		iHostTrainable = TrainNNThresh(rSes, false);
	}
	printf("%d trainable sample outputs found on host.\n", iHostTrainable);
		cudaMemcpy(MinWts, rSes.rNet->Wt, 16*2048, cudaMemcpyHostToHost);
	if(iTrainable-iHostTrainable) printf("\nTRAINABLE FAIL\n\n");
	else printf("\nTRAINABLE PASS\n\n");

	printf("\n\nBEGIN BACKPROP TEST\n\n");
	// begin backpropagation test HOST
	{
		boost::timer::auto_cpu_timer t;
		cuBackpropSingleSample(rSes, 2, *rSes.rNet, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rNet->Deltas, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval);
	}
		cudaMemcpy(rSes.rNet->Wt, MinWts, 16*2048, cudaMemcpyHostToHost);
	// begin backpropagation test DEVICE
	Team->LetTrainNNThresh(rSes, 2, 1, 'S', rSes.dTargetRMSE, rSes.iEpochLength);
	
	Barge->LetWriteWeights(rSes);// record weights for posterity
	
	if (rSes.iWarnings) fprintf(stderr, "%d warnings.\n", rSes.iWarnings);
	if (rSes.iErrors) fprintf(stderr, "%d operational errors.\n", rSes.iErrors);

	return iReturn;
}


long cDrover::AskSampleSetName(struct rohanContext& rSes)
{mIDfunc /// chooses the learning set to be worked with Ante-Loop
	int iReturn=0; 
	//rSes.rLearn->bContInputs=false;
	//rSes.rLearn->iContOutputs=(int)false;
	cout << "Samples treated as discrete or continuous by fractionality. XX" << endl;

	printf("Enter 0 for 10K-set, weights\n\t 1 for 3-set, 331 weights\n\t 2 for 150-set, no wgts\n\t 3 for 4-set, 2x21 wgts\n\t 4 for 2-1 rand weights");
	printf("\n\t 5 for 416 samples x 200 inputs\nEnter 10+ for basic diag\n\t30+ for more diag\n\t70+ for full diag\n");
	std::cin >> iDebugLvl;
	switch ( iDebugLvl % 10) {
		case 0:
		  rSes.rLearn->sLearnSet="AirplanePsDN1W3S10k.txt";
		  break;
		case 1:
		  rSes.rLearn->sLearnSet="trivial3331.txt";
		  break;
		case 2:
		  rSes.rLearn->sLearnSet="iris.txt";
		  break;
		case 3:
		  rSes.rLearn->sLearnSet="trivial221.txt";
		  break;
		case 4:
		  rSes.rLearn->sLearnSet="trivial3.txt";	
		  break;
		case 5:
		  rSes.rLearn->sLearnSet="PC-63-32-200-LearnSet.txt";
		  break;
		default:
		  rSes.rLearn->sLearnSet="iris.txt";
		  break;
	}
	rSes.iDebugLvl=iDebugLvl/=10; // drop final digit
	if (iDebugLvl) fprintf(stderr, "Debug level is %d.\n", iDebugLvl);
	return iReturn;
}


long cDrover::DoMainLoop(struct rohanContext& rSes)
{mIDfunc /// Trains a weight set to more closely reproduce the sampled outputs from the sampled inputs, and other options.
	int iReturn=0, iSelect=1;
	cout << "Main duty loop begin." << endl;
	
	while(iSelect){
		//Team->LetSlack(rSes);
		//Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
		//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
		iSelect=DisplayMenu(0, rSes);
		if (iSelect==1) iReturn=BeginSession(rSes); // new or resume session
		if (iSelect==2) iReturn=GetNNTop(rSes);
		//if (iSelect==3) iReturn=ReGetSampleSet(rSes); XX
		if (iSelect==4) iReturn=GetWeightSet(rSes);
		if (iSelect==5) iReturn=this->LetInteractiveEvaluation(rSes);
		if (iSelect==6) iReturn=this->LetInteractiveLearning(rSes);
		if (iSelect==7) iReturn=cuPreSaveNNWeights(rSes);
		if (iSelect==8) {iReturn=cuRandomizeWeights(rSes); 
						Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
						rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
		}
		if (iSelect==9) {
			char afname[100]="", bfname[100]="", 
				lineIn[255], *tok, *cSample;
			FILE * ASCIN, * BINOUT; cuDoubleComplex way;
			cout << "Enter name of .txt file to convert to .wgt:" << endl;
			cin >> bfname;
			strcat(afname, bfname); strcat(afname, ".txt"); strcat(bfname, ".wgt");
			AsciiFileHandleRead( afname, &ASCIN );
			BinaryFileHandleWrite( bfname, &BINOUT );
			#define MAX_REC_LEN 65536 /* Maximum size of input buffer */
			while(fgets(lineIn, MAX_REC_LEN, ASCIN)) { //each line is read in turn
				cSample = _strdup(lineIn);
				printf("%s", cSample); 
				tok=strtok( cSample , " ,\t" ); way.x=atof( tok ); 
				tok=strtok( NULL, " ,\t" ); way.y=atof( tok );
				printf("%f+%f\n", way.x, way.y);
				fwrite( &(way.x) , sizeof(double), 1, BINOUT);
				fwrite( &(way.y) , sizeof(double), 1, BINOUT);
			}
			fclose(ASCIN);
			fclose(BINOUT);
		}
		//SilentDiagnostics(rSes);
	}
	return 0;
}


int DisplayMenu(int iMenuNum, struct rohanContext& rSes)
{mIDfunc
	char a='.';
	if(iMenuNum==0){
		printf("\n1 - Label session \"%s\"", rSes.sSesName);
		printf("\n2 - Network topology setup");
		printf("\n3 - Sample set load");
		printf("\n4 - Weight set load");
		printf("\n5 - Evaluation Feed Forward");
		printf("\n6 / Learning");
		printf("\n7 - Save weights");
		printf("\n8 - Randomize weights");
		printf("\n9 - utilities");
		printf("\n0 - Quit");
		printf("\n %s %d samples MAX %f, %d trainable", rSes.rLearn->sLearnSet, rSes.rLearn->lSampleQty, rSes.dMAX, TrainNNThresh(rSes, false));
		printf("\nDRMSE %f YRMSE %f ", rSes.dTargetRMSE, rSes.dRMSE);
		for(int i=0;i<rSes.rNet->iLayerQty;++i)
			printf("L%d %d; ", i, rSes.rNet->rLayer[i].iNeuronQty);
		printf("%d sectors ", rSes.rNet->iSectorQty);
		if (rSes.bRInJMode) printf("ReverseInput "); 
	}
	if(iMenuNum==50){
		printf("\n1 - Include inputs: %d", rSes.iSaveInputs);
		printf("\n2 - Include outputs: %d", rSes.iSaveOutputs);
		printf("\n3 - Include sample index: %d", rSes.iSaveSampleIndex);
		printf("\n4 - change Samples used: %d", rSes.lSampleQtyReq);
		printf("\n5 X Synchronous commence");
		printf("\n6 / Asynchronous commence");
		printf("\n7 - change Blocks per kernel: %d", rSes.iEvalBlocks);
		printf("\n8 - change Threads per block: %d", rSes.iEvalThreads);
		printf("\n9 - Save evals");
		printf("\n0 - Quit");
		printf("\n %s %d samples MAX %f, %d trainable", rSes.rLearn->sLearnSet, rSes.rLearn->lSampleQty, rSes.dMAX, TrainNNThresh(rSes, false));
		printf("\nDRMSE %f YRMSE %f ", rSes.dTargetRMSE, rSes.dRMSE);
		for(int i=0;i<rSes.rNet->iLayerQty;++i)
			printf("L%d %d; ", i, rSes.rNet->rLayer[i].iNeuronQty);
		printf("%d sectors ", rSes.rNet->iSectorQty);
		if (rSes.bRInJMode) printf("ReverseInput "); 
	}
	if(iMenuNum==60){
		printf("\n1 - change Target RMSE: % #3.3g", rSes.dTargetRMSE);
		printf("\n2 - change MAX error: % #3.3g", rSes.dMAX);
		printf("\n3 - change Epoch length: %d", rSes.iEpochLength);
		printf("\n4 - change Samples used: %d", rSes.lSampleQtyReq);
		printf("\n5 X Synchronous commence");
		printf("\n6 / Asynchronous commence");
		printf("\n7 - change Blocks per kernel: %d", rSes.iBpropBlocks);
		printf("\n8 - change Threads per block: %d", rSes.iBpropThreads);

		//printf("\n9 - change status: %s %s", (Team->GetHitched() ? "HITCHED":"UNHITCHED"),
		//	(Team->GetTaut() ? "TAUT":"SLACK"));
		printf("\n0 - Quit");
		printf("\n %s %d samples MAX %f, %d trainable", rSes.rLearn->sLearnSet, rSes.rLearn->lSampleQty, rSes.dMAX, TrainNNThresh(rSes, false));
		printf("\nDRMSE %f YRMSE %f ", rSes.dTargetRMSE, rSes.dRMSE);
		for(int i=0;i<rSes.rNet->iLayerQty;++i)
			printf("L%d %d; ", i, rSes.rNet->rLayer[i].iNeuronQty);
		printf("%d sectors ", rSes.rNet->iSectorQty);
		if (rSes.bRInJMode) printf("ReverseInput "); 
	}
	printf("\n");
	// http://www.cplusplus.com/doc/ascii/
	while(a<'0'||a>'9')
		a=_getch();
	return ((int)a)-48;
}


int BeginSession(struct rohanContext& rSes)
{mIDfunc /// accepts keyboard input to define the name of the session, which will be used to name certain output files.
	cout << "\nEnter a session name: ";
	cin >> rSes.sSesName; 

	return 1;
}


int GetNNTop(struct rohanContext& rSes)
{mIDfunc /// sets up network poperties and data structures for use
	char sNeuronsPerLayer[254];
	int iSectorQty, iInputQty;

	cout << "Enter # of sectors (0 to return): ";
	cin >> iSectorQty;
	if(iSectorQty){
		cout << "Enter # of inputs (0 to return): ";
		cin >> iInputQty; // last chance to quit
	}
	if(iSectorQty && iInputQty) {
		cuFreeNNTop(rSes); // release old network structures
		rSes.rNet->iSectorQty=iSectorQty; // update sector qty
		rSes.rNet->kdiv2=iSectorQty/2; // update sector qty
		rSes.rLearn->iInputQty=iInputQty; // upsdate input qty
		cout << "Enter numbers of neurons per layer separated by commas, \ne.g. 63,18,1 : ";
		cin >> sNeuronsPerLayer;
		cuMakeLayers(iInputQty, sNeuronsPerLayer, rSes); // make new layers
		rSes.rNet->dK_DIV_TWO_PI = rSes.rNet->iSectorQty / TWO_PI; // Prevents redundant conversion operations
		cuMakeNNStructures(rSes); // allocates memory and populates network structural arrays
		cuRandomizeWeights(rSes); // populate newtork with random weight values
		printf("Random weights loaded.\n");
		printf("%d-valued logic sector table made.\n", cuSectorTableMake(rSes));
		printf("\n");
		return rSes.rNet->iLayerQty;
	}
	else
		return 999;
}


long cuFreeNNTop(struct rohanContext &rSes)
{mIDfunc/// frees data structures related to network topology
//	cublasStatus csStatus;
	
	free( rSes.rNet->cdcSectorBdry );
	// layer components
	free( rSes.rNet->rLayer[0].ZOutputs ); // Layer Zero has no need of weights!
	//csStatus = cublasFree( rSes.rNet->rLayer[0].ZOutputs ); // de-allocate a GPU-space pointer to a vector of complex neuron outputs for each layer
	//mCuMsg(csStatus,"cublasFree()")
	
	for (int i=1; i < rSes.rNet->iLayerQty; ++i){ 
		struct rohanLayer& lay=rSes.rNet->rLayer[i];
		free( lay.Weights ); // de-allocate a pointer to an array of arrays of weights
		free( lay.ZOutputs ); // de-allocate a pointer to an array of arrays of weights
		free( lay.Deltas ); // free the backprop areas
	}
	free( rSes.rNet->rLayer ); // free empty layers
	printf("Network structures freed.\n");
	return 0;
}

int cuMakeNNStructures(struct rohanContext &rSes)
{mIDfunc
/*! Initializes a neural network structure of the given number of layers and
 *  layer populations, allocates memory, and populates the set of weight values randomly.
 *
 * iLayerQty = 3 means Layer 1 and Layer 2 are "full" neurons, with output-only neurons on layer 0.
 * 0th neuron on each layer is a stub with no inputs and output is alawys 1+0i, to accomodate internal weights of next layer.
 * This allows values to be efficiently calculated by referring to all layers and neurons identically.
 * 
 * rLayer[1].iNeuronQty is # of neurons in Layer 1, not including 0
 * rLayer[2].iNeuronQty is # of neurons in Layer 2, not including 0
 * rLayer[0].iNeuronQty is # of inputs in Layer 0 
 * iNeuronQTY[1] is # of neurons in Layer 1, including 0
 * iNeuronQTY[2] is # of neurons in Layer 2, including 0 */

	long lReturn=0;
const cuDoubleComplex cdcZero = { 0, 0 }, cdcInit = { -999.0, 999.0 };
	//cdcInit.x=-999.0; cdcInit.y=999.0;
	for (int i=0; i < rSes.rNet->iLayerQty; ++i){  //Layer Zero has no need of weights! 8/13/2010
		struct rohanLayer& lay = rSes.rNet->rLayer[i];
		struct rohanNetwork * rnSrc = rSes.rNet;
		long DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE, L=i;
		//setup dimension values
		DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
		DSIZE = DQTY * sizeof(cuDoubleComplex) ;
		NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
		NSIZE = NQTY * sizeof(cuDoubleComplex) ;
		WQTY = DQTY * NQTY ; // weights = neurons * dendrites
		WSIZE = WQTY * sizeof(cuDoubleComplex) ;
		
		//allocate memory
		lay.Weights = (cuDoubleComplex*)malloc ( WSIZE ); // 2D array of complex weights
			mCheckMallocWorked(lay.Weights)
		lay.XInputs = (cuDoubleComplex*)malloc( DSIZE ); //allocate a pointer to an array of outputs
			mCheckMallocWorked(lay.XInputs)
		lay.ZOutputs = (cuDoubleComplex*)malloc( NSIZE ); //allocate a pointer to an array of outputs
			mCheckMallocWorked(lay.ZOutputs)
		lay.Deltas = (cuDoubleComplex*)malloc( NSIZE ); //allocate a pointer to a parallel array of learned corrections
			mCheckMallocWorked(lay.Deltas)
		lReturn+=lay.iNeuronQty*lay.iDendriteQty;
   		lReturn+=lay.iNeuronQty;
	
		//init values
		for (int i=0; i <= lay.iDendriteQty; ++i){
			for (int k=0; k <= lay.iNeuronQty; ++k){ 
				lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)].x=(double)rand()/65535; // necessary to promote one operand to double to get a double result
				lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)].y=(double)rand()/65535;
				//lay.Deltas[IDX2C(i, k, lay.iDendriteQty+1)]=cdcInit;
			}
			// reset neuron 0 weights to null
			lay.Weights[IDX2C(i, 0, lay.iDendriteQty+1)] = cdcZero;
			// mark inputs as yet-unused
			lay.XInputs[i]=cdcInit;
		}
		lay.Weights[IDX2C(0, 0, lay.iDendriteQty+1)].x=1.0; // neuron 0, dendrite 0 interior weight should always be equal to 1+0i
		for (int k=0; k <= lay.iNeuronQty; ++k){
			// mark outputs and deltas as yet-unused
			lay.ZOutputs[k]=cdcInit;
			lay.Deltas[k]=cdcInit;
		}
	}
	return lReturn; //return how many weights and outputs allocated
}


int GetWeightSet(struct rohanContext& rSes)
{mIDfunc /// chooses and loads the weight set to be worked with
	int iReturn=0; 
	char sWeightSet[254];
	FILE *fileInput;
	
	cout << "Enter name of binary weight set: ";
	std::cin >> sWeightSet;
	// File handle for input
	
	iReturn=BinaryFileHandleRead(sWeightSet, &fileInput);
	if (iReturn==0) // unable to open file
		++rSes.iErrors;
	else{ // file opened normally
		// file opening and reading are separated to allow for streams to be added later
		iReturn=cuNNLoadWeights(rSes, fileInput);
		if (iReturn) printf("%d weights read.\n", 
			iReturn);
		else {
			printf("No Weights Read\n");
			iReturn=0;
		}
	}
	printf("\n");
	return iReturn;
}


long cDrover::LetInteractiveEvaluation(struct rohanContext& rSes)
{mIDfunc /// allows user to ask for different number of samples to be evaluated
	int iReturn=0, iSelect=1;
	
	while(iSelect){
		//Team->LetSlack(rSes);
		//Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
		//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
		iSelect=DisplayMenu(50, rSes);
		if (iSelect==1) {rSes.iSaveInputs=(rSes.iSaveInputs ? false: true); }
		if (iSelect==2) {rSes.iSaveOutputs=(rSes.iSaveOutputs ? false: true); }
		if (iSelect==3) {rSes.iSaveSampleIndex=(rSes.iSaveSampleIndex ? false: true); }
		if (iSelect==4) {printf("Enter samples requested\n");std::cin >> rSes.lSampleQtyReq;} //
		//if (iSelect==5) {} // synchronous kernel launch
		if (iSelect==6) { // asynchronous kernel launch
					++iReturn;
					// serial values are computed and then displayed
					//cuEvalNNLearnSet(rSes);
					Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
					rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
					printf("%s: first %d samples requested\nRMSE= %f", rSes.rNet->sWeightSet, rSes.lSampleQtyReq, rSes.dRMSE);
					double devdRMSE=knlRMSEopt(rSes, rSes.lSampleQtyReq, 1, 'R'); // XX
					printf("/%f\n", devdRMSE);
		}
		if (iSelect==7) {printf("Enter blocks per kernel\n");std::cin >> rSes.iEvalBlocks;}
		if (iSelect==8) {printf("Enter threads per block\n");std::cin >> rSes.iEvalThreads;}
		if (iSelect==9) {Barge->LetWriteEvals(rSes, *rSes.rLearn);} 
		if (iSelect==0) {} // quit
	}
	return iReturn;
}


long cDrover::LetInteractiveLearning(struct rohanContext& rSes)
{mIDfunc /// allows user to select learning thresholds
	int iReturn=0, iSelect=1;
	
	while(iSelect){
		//Team->LetSlack(rSes);
		//Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
		//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
		iSelect=DisplayMenu(60, rSes);
		if (iSelect==1) {printf("Enter desired RMSE for learning\n");std::cin >> rSes.dTargetRMSE;}
		if (iSelect==2) {printf("Enter MAX allowable error per sample\n");std::cin >> rSes.dMAX;}
		if (iSelect==3) {printf("Enter iterations per epoch\n");std::cin >> rSes.iEpochLength;}
		if (iSelect==4) {printf("Enter samples requested\n");std::cin >> rSes.lSampleQtyReq;} //
		//if (iSelect==5) {} // synchronous kernel launch
		if (iSelect==6) { // asynchronous kernel launch
					++iReturn;
							Team->LetTaut(rSes);
							//rSes.lSamplesTrainable=knlBackProp( rSes, rSes.lSampleQtyReq, 1, 'R');
							rSes.lSamplesTrainable=Team->LetTrainNNThresh( rSes, rSes.lSampleQtyReq, 1, 'R', rSes.dTargetRMSE, rSes.iEpochLength);
	}
		if (iSelect==7) {printf("Enter blocks per kernel\n");std::cin >> rSes.iBpropBlocks;}
		if (iSelect==8) {printf("Enter threads per block\n");std::cin >> rSes.iBpropThreads;}
		//if (iSelect==9) {} //
		if (iSelect==0) {} // quit
	}
	return iReturn;
}


long cDrover::DoPostLoop(struct rohanContext& rSes) 
{mIDfunc /// Final operations including freeing of dynamically allocated memory are called from here. 
	int iReturn=0, iSelect=0;

	printf("Program terminated after %d warning(s), %d operational error(s).\n", rSes.iWarnings, rSes.iErrors);
	DoEndItAll(rSes);
	printf("Waiting on keystroke...\n");
	_getch();

	return 0;
}


int SilentDiagnostics(struct rohanContext& rSes)
{mIDfunc /// show some statistics, dump weights, and display warning and error counts
	int iReturn=0;
	
	long lSamplesEvald = cuEvalNNLearnSet(rSes);
	int iDifferent = OutputValidate(rSes);
	//printf("%d differences found on verify.\n", iDifferent);
	rSes.dRMSE = RmseNN(rSes, 0);
	//printf("RMSE %f\n", rSes.dRMSE);
	// some illustrative default values
	rSes.dTargetRMSE=floor(rSes.dRMSE-1.0)+1.0;
	rSes.dMAX=(double)abs(rSes.dRMSE-1.0);
	//printf("RMSE target is %f.\n", rSes.dTargetRMSE);
	//printf("MAX threshold is %f.\n", rSes.dMAX);
	int iTrainable = TrainNNThresh(rSes, false);
	//printf("%d trainable sample outputs found.\n", iTrainable);

	// dump weights for verification
	FILE *fileOutput; // File handle for output
	iReturn=AsciiFileHandleWrite("weightdump.txt", &fileOutput);
	AsciiWeightDump(rSes, fileOutput); //XX link error
	
	if (rSes.iWarnings) fprintf(stderr, "%d warnings.\n", rSes.iWarnings);
	if (rSes.iErrors) fprintf(stderr, "%d operational errors.\n", rSes.iErrors);

	return iReturn;
}


long cDrover::DoEndItAll(struct rohanContext& rSes)
{mIDfunc /// prepares for graceful ending of program
	int iReturn=0;

	Team->LetUnHitch(rSes);
	iReturn=Barge->DoCuFree(rSes);
	
	return iReturn;
}
