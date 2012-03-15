/* Includes, cuda */

#include "Rohan.h"
#include "Rohan-kernel.h"
#include "rohan-io.h"
//#include "cuPrintf1.cuh"
#include <conio.h> //for _getch
#include <iostream>
#include <iomanip>
using namespace std;
#include "complex-math-func.h"
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))
extern int iDebugLvl, iTrace;

int ShowMeDiffWeights(struct rohanContext& rSes)
{ long diffs=0;
	for (int j=1; j < rSes.rNet->iLayerQty; ++j){ //no weights for layer 0
		struct rohanLayer& lay = rSes.rNet->rLayer[j];
		for (int k=1; k <= lay.iNeuronQty; ++k){ // no weights for neuron 0
			for (int i=0; i <= lay.iDendriteQty; ++i){
				cuDoubleComplex way = rSes.rNet->Wt[IDX2C(i+rSes.rNet->iWeightOfst[j], k, rSes.rNet->iDendrtQTY[j])] ;
				cuDoubleComplex oway = rSes.rNet->rLayer[j].Weights[IDX2C( i, k, rSes.rNet->rLayer[j].iDendriteQty+1 )] ;
				if(way.x-oway.x + way.y-oway.y)
					printf("L%d N%d D%d %f+%f\n", j, k, i, way.x-oway.x, way.y-oway.y, ++diffs);
			}
		}
	}
	return diffs;
}

int ShowMeDiffSums(struct rohanNetwork& rNet, cuDoubleComplex * Sums, FILE * fShow, char cSymbol, int x1, int x2, int x3)
{/// checks all elements of Sums against their corresponding XInputs and ZOutputs
	for(int L=1; L<rNet.iLayerQty; ++L){
		for (int i=0; i<=rNet.rLayer[L].iNeuronQty; ++i){
			double X = rNet.rLayer[L].ZOutputs[i].x - Sums[ rNet.iNeuronOfst[L]+i ].x;
			double Y = rNet.rLayer[L].ZOutputs[i].y - Sums[ rNet.iNeuronOfst[L]+i ].y;
			cuDoubleComplex delta = { X, Y };
			double Z = CxAbs(delta);
			if (Z>0.01){
				fprintf(fShow, "host%c ZOutput %d,%d Sums %d position %d,%d,%d = %f,%f\n", cSymbol, L, i, rNet.iNeuronOfst[L]+1, x1, x2, x3, X+Y, Z);
				Sums[ rNet.iNeuronOfst[L]+i ] = rNet.rLayer[L].ZOutputs[i] ;
			}
		}
	}
	return 0;
}

int ShowMeResetSums(struct rohanNetwork& rNet, cuDoubleComplex * Sums)
{/// checks all elements of Sums against their corresponding XInputs and ZOutputs
	for(int L=0; L<rNet.iLayerQty; ++L){
		for (int i=0; i<=rNet.rLayer[L].iNeuronQty; ++i){
			Sums[ rNet.iNeuronOfst[L]+i ] = rNet.rLayer[L].ZOutputs[i] ;
		}
	}
	return 0;
}


int ShowMeSes(struct rohanContext& rSes, int iMode)
{/// displays labeled parts of session context and associated values
	int iReturn=0;
	FILE *fShow;
	if (iMode)
		fShow=rSes.debugHandle;
	else
		fShow=stdout;


	fprintf(fShow, "\n== SHOWME Session context ==\n");
	//printf("Samples requested: %d\n", rSes.lSampleQtyReq); // size of working subset of samples
	//printf("Target RMSE: %e\n", rSes.dTargetRMSE); // target RMSE
	//printf("MAX threshold %e\n", rSes.dMAX); // dont backprop unless error is this much
	//printf("Continuous Activation %d\n", rSes.bContActivation); // use Continuous activation function
	//printf("Last RMSE: %e\n", rSes.dRMSE); // evaluated RMSE

	fprintf(fShow, "iDebugLvl %d\n", rSes.iDebugLvl ); 
	fprintf(fShow, "iEvalMode %d\n", rSes.iEvalMode ); 
	fprintf(fShow, "iWarnings %d\n", rSes.iWarnings ); 
	fprintf(fShow, "iErrors %d\n", rSes.iErrors ); 

	fprintf(fShow, "bCLargsUsed %d\n", rSes.bCLargsUsed ); 
	fprintf(fShow, "bConfigFileUsed %d\n", rSes.bConfigFileUsed ); 
	fprintf(fShow, "bConsoleUsed %d\n", rSes.bConsoleUsed ); 
	fprintf(fShow, "bContActivation %d\n", rSes.bContActivation); 
	fprintf(fShow, "bCUDAavailable %d\n", rSes.bCUDAavailable ); 
	fprintf(fShow, "bLearnSetSpecUsed %d\n", rSes.bLearnSetSpecUsed ); 
	fprintf(fShow, "bRInJMode %d\n", rSes.bRInJMode ); 
	
	fprintf(fShow, "bRMSEon %d\n", rSes.bRMSEon ); 
	fprintf(fShow, "dMAX %e\n", rSes.dMAX ); 
	fprintf(fShow, "dRMSE %e\n", rSes.dRMSE ); 
	fprintf(fShow, "dTargetRMSE %e\n", rSes.dTargetRMSE ); 
	fprintf(fShow, "iEpochLength %d\n", rSes.iEpochLength ); 
	fprintf(fShow, "lSampleQtyReq %d\n", rSes.lSampleQtyReq ); 
	fprintf(fShow, "lSamplesTrainable %d\n", rSes.lSamplesTrainable ); 

	fprintf(fShow, "sConfigFileOpt %s\n", rSes.sConfigFileOpt );
	fprintf(fShow, "sCLargsOpt %s\n", rSes.sCLargsOpt );
	fprintf(fShow, "sLearnSetSpecOpt %s\n", rSes.sLearnSetSpecOpt );
	fprintf(fShow, "sConsoleOpt %s\n", rSes.sConsoleOpt );
	fprintf(fShow, "sSesName %s\n", rSes.sSesName );

	return iReturn;
}

int ShowMeLS(struct rohanContext& rSes, int iMode)
{/// displays labeled contents of rLearn
	int iReturn=0;
	int iPrintFlag=0;
	FILE *fShow;
	//cublasStatus csStatus;
	if (iMode)
		fShow=rSes.debugHandle;
	else
		fShow=stdout;

	fprintf(fShow,"=== SHOWME Learning Set: %s ===\n", rSes.rLearn->sLearnSet); 
	fprintf(fShow,"Samples %d, Sectors %d, Values/Line %d = Inputs %d + Outputs %d\n", rSes.rLearn->lSampleQty, rSes.rNet->iSectorQty, rSes.rLearn->iValuesPerLine, rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
	// 0 is ContActivation outputs, 1 is discrete outputs
	if(rSes.rLearn->iEvalMode)
		fprintf(fShow,"Eval Mode %d Discrete outputs.\n", rSes.rLearn->iEvalMode);
	else
		fprintf(fShow,"Eval Mode %d Continuous outputs.\n", rSes.rLearn->iEvalMode);
	fprintf(fShow,"lSampleIdxReq %d\n", rSes.rLearn->lSampleIdxReq);
	fprintf(fShow,"bContInputs %d\n", rSes.rLearn->bContInputs);
	fprintf(fShow,"iContOutputs %d\n", rSes.rLearn->iContOutputs);
	fprintf(fShow,"sLearnSet %s\n", rSes.rLearn->sLearnSet);
	//commence with listing values
	for(long s=0; s<rSes.rLearn->lSampleQty; ++s){
		fprintf( fShow, "%dX|", s);
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i)
			fprintf( fShow, "%f\t", rSes.rLearn->dXInputs[ IDX2C( i, s, rSes.rLearn->iInputQty+1 ) ] );
		fprintf( fShow, "\n%dD|", s);
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i)
			fprintf( fShow, "%f\t", rSes.rLearn->dDOutputs[ IDX2C(  i, s, rSes.rLearn->iOutputQty+1 ) ] );
		fprintf( fShow, "\n");
	}
	
	return iReturn;
}

int ShowMeLay(struct rohanContext& rSes, int iMode)
{/// displays sector table
	int iReturn=0;
	FILE *fShow;
	//cublasStatus csStatus;
	if (iMode)
		fShow=rSes.debugHandle;
	else
		fShow=stdout;
	fprintf(fShow,"=== SHOWME Layers ===\n"); 
	for(int L=1; L<rSes.rNet->iLayerQty; ++L){
		fprintf(fShow, "\n%dX|", L);
		for (int d=0; d<=rSes.rNet->rLayer[L].iDendriteQty; ++d)
			fprintf(fShow, "%f+%f,%d ", rSes.rNet->rLayer[L].XInputs[d].x, rSes.rNet->rLayer[L].XInputs[d].y, d);
		fprintf(fShow, "\n%dZ|", L);
		for (int n=0; n<=rSes.rNet->rLayer[L].iNeuronQty; ++n)
			fprintf(fShow, "%f+%f,%d ", rSes.rNet->rLayer[L].ZOutputs[n].x, rSes.rNet->rLayer[L].ZOutputs[n].y, n);
	}
	fprintf(fShow, "\n");
	return iReturn;
}

int ShowMeEvals(struct rohanContext& rSes, int iMode)
{/// displays labeled contents of rLearn
	int iReturn=0;
	int iPrintFlag=0;
	FILE *fShow;
	//cublasStatus csStatus;
	if (iMode)
		fShow=rSes.debugHandle;
	else
		fShow=stdout;

	fprintf(fShow,"=== SHOWME Evals for: %s ===\n", rSes.rLearn->sLearnSet); 
	fprintf(fShow,"Samples %d, Sectors %d, Values/Line %d = Inputs %d + Outputs %d\n", rSes.rLearn->lSampleQty, rSes.rNet->iSectorQty, rSes.rLearn->iValuesPerLine, rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
	// 0 is ContActivation outputs, 1 is discrete outputs
	if(rSes.rLearn->iEvalMode)
		fprintf(fShow,"Eval Mode %d Discrete outputs.\n", rSes.rLearn->iEvalMode);
	else
		fprintf(fShow,"Eval Mode %d Continuous outputs.\n", rSes.rLearn->iEvalMode);
	fprintf(fShow,"lSampleIdxReq %d\n", rSes.rLearn->lSampleIdxReq);
	fprintf(fShow,"bContInputs %d\n", rSes.rLearn->bContInputs);
	fprintf(fShow,"iContOutputs %d\n", rSes.rLearn->iContOutputs);
	fprintf(fShow,"sLearnSet %s\n", rSes.rLearn->sLearnSet);
	//commence with listing values
	for(long s=0; s<rSes.rLearn->lSampleQty; s+=1000){
		fprintf( fShow, "%dD|", s);
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i)
			fprintf( fShow, "%f\t", rSes.rLearn->dDOutputs[ IDX2C(  i, s, rSes.rLearn->iOutputQty+1 ) ] );
		fprintf( fShow, "\n%dY|", s);
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i)
			fprintf( fShow, "%f\t", rSes.rLearn->dYEval[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ] );
		fprintf( fShow, "\n%dA|", s);
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i)
			fprintf( fShow, "%f\t", rSes.rLearn->dAltYEval[ IDX2C(  i, s, rSes.rLearn->iOutputQty+1 ) ] );
		fprintf( fShow, "\n");
	}
	/*for(long s=0; s<rSes.rLearn->lSampleQty; s+=1000){
		fprintf( fShow, "%dD|", s);
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){
			fprintf( fShow, "%f,%g\t", rSes.rLearn->dDOutputs[ IDX2C(  i, s, rSes.rLearn->iOutputQty+1 ) ], 
				abs(rSes.rLearn->dYEval[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ]-rSes.rLearn->dAltYEval[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ]) );
		}
		fprintf( fShow, "\n%dY|", s);
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){
			fprintf( fShow, "%f,%g\t", rSes.rLearn->dYEval[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ],
				pow(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ]-rSes.rLearn->dYEval[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ], 2.0) );
		}
		fprintf( fShow, "\n%dA|", s);
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){
			fprintf( fShow, "%f,%g\t", rSes.rLearn->dAltYEval[ IDX2C(  i, s, rSes.rLearn->iOutputQty+1 ) ],
				pow(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ]-rSes.rLearn->dAltYEval[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ], 2.0) );
		}
		fprintf( fShow, "\n");
	}*/
	return iReturn;
}

//int TestPatLS(struct rohanContext& rSes, long bWait)
//{/// fills Learn Set and associated values with known values, transfers to GPU mem and back to check for fidelity
//	//commence with setting values
//	cublasStatus csStatus;
//	int iReturn=0;
//	//struct rohanLayer& LZero = rSes.rNet->rLayer[0];
//	//struct rohanLayer& LTop = rSes.rNet->rLayer[rSes.rNet->iLayerQty-1];
//	for(long s=0; s<rSes.rLearn->lSampleQty; ++s){
//		struct rohanSample& sam = rSes.rLearn->rSample[s];
//
//		for (int i=0; i<=rSes.rLearn->iInputQty; ++i){ //displaying complex-converted values for inputs
//			sam.cdcXInputs[i].x = 0+((s+1)*100.0 + i*10.0 + 1.0);
//			sam.cdcXInputs[i].y = 0-((s+1)*100.0 + i*10.0 + 1.0);
//		}
//		csStatus=cublasSetVector(rSes.rLearn->iInputQty+1, sizeof(cuDoubleComplex), sam.cdcXInputs, 1, sam.gpuXInputs, 1);
//			mCuMsg(csStatus,"cublasSetVector()") // copies from CPU to GPU
//		csStatus=cublasGetVector(rSes.rLearn->iInputQty+1, sizeof(cuDoubleComplex), sam.gpuXInputs, 1, sam.cdcAltXInputs, 1);
//			mCuMsg(csStatus,"cublasGetVector()") // and copies back to CPU again for comparison
//
//		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ //displaying complex-converted values for outputs
//			sam.cdcDOutputs[i].x = 0+((s+1)*100.0 + i*10.0);
//			sam.cdcDOutputs[i].y = 0-((s+1)*100.0 + i*10.0);
//		}
//		csStatus=cublasSetVector(rSes.rLearn->iOutputQty+1, sizeof(cuDoubleComplex), sam.cdcDOutputs, 1, sam.gpuDOutputs, 1);
//			mCuMsg(csStatus,"cublasSetVector()") // copies from CPU to GPU
//		csStatus=cublasGetVector(rSes.rLearn->iOutputQty+1, sizeof(cuDoubleComplex), sam.gpuDOutputs, 1, sam.cdcAltDOutputs, 1);
//			mCuMsg(csStatus,"cublasGetVector()") // and copies back to CPU again for comparison
//	}
//	if (bWait) // wait on keystroke or not
//		{ mExitKeystroke }
//	return iReturn;
//}


//int ShowMeSam(struct rohanContext& rSes, long S, long bWait)
//{///displays a particular sample
//	int iReturn=0;
//	int iPrintFlag=0;
//	cublasStatus csStatus;
//
//	printf("SHOWME Sample %d, ", S); 
//	if(rSes.rLearn->iEvalMode)
//		printf("Discrete outputs.\n", rSes.rLearn->iEvalMode);
//	else
//		printf("Continuous outputs.\n", rSes.rLearn->iEvalMode);
//	//commence with listing values
//	cout << fixed;
//	struct rohanSample& sam = rSes.rLearn->rSample[S];
//	csStatus=cublasGetVector(rSes.rLearn->iInputQty+1, sizeof(cuDoubleComplex), sam.gpuXInputs, 1, sam.cdcAltXInputs, 1);
//		mCuMsg(csStatus,"cublasGetVector()") // Network inputs transferred form GPU to host
//	csStatus=cublasGetVector(rSes.rLearn->iOutputQty+1, sizeof(cuDoubleComplex), sam.gpuYEval, 1, sam.cdcAltYEval, 1);
//		mCuMsg(csStatus,"cublasGetVector()") // cx yielded outputs transferred
//	csStatus=cublasGetVector(rSes.rLearn->iOutputQty+1, sizeof(cuDoubleComplex), sam.gpudYEval, 1, sam.cdcAltDOutputs, 1);
//		mCuMsg(csStatus,"cublasGetVector()") // scalar yielded outputs transferred, temporarily to AltDOutputs
//	for (int i=0; i<=rSes.rLearn->iOutputQty; ++i)
//		sam.dAltYEval[i] = sam.cdcAltDOutputs[i].x; // scalar values copied to their final resting place 
//	csStatus=cublasGetVector(rSes.rLearn->iOutputQty+1, sizeof(cuDoubleComplex), sam.gpuDOutputs, 1, sam.cdcAltDOutputs, 1);
//		mCuMsg(csStatus,"cublasGetVector()") // desired complex outputs transferred
//
//	// begin display routines
//	cout << setw(4) << S << ":" << setprecision(2);
//	for (int i=0; i<=rSes.rLearn->iValuesPerLine; ++i){
//		cout << " " << setw(3) << sam.dXInputs[i];
//		if (i==rSes.rLearn->iInputQty)
//			cout << "\tO:"; // outputs begin here
//		if (i==(rSes.rLearn->iInputQty+rSes.rLearn->iOutputQty) && i<rSes.rLearn->iValuesPerLine)
//			cout << "\tX:"; // excess values begin here
//	}
//	cout << "\n" << setprecision(6);
//	for (int i=0; i<=rSes.rLearn->iInputQty; ++i){ //displaying complex-converted values for inputs
//		cout << S << "/" << i << "ITp " << setw(9) << sam.cdcXInputs[i].x << "\t" << setw(9) << sam.cdcXInputs[i].y << "i\t"; 
//			if((abs(sam.cdcAltXInputs[i].x-sam.cdcXInputs[i].x)+abs(sam.cdcAltXInputs[i].y-sam.cdcXInputs[i].y))>0.0000005) cout << ">" << ++iReturn << ">";
//		cout << "~ " << setw(9) << sam.cdcAltXInputs[i].x << "\t" << setw(9) << sam.cdcAltXInputs[i].y << "i\n"; 
//	}
//	cout << "   Y:" << setprecision(2); // feed forward evaluation
//	for (int i=0; i<=rSes.rLearn->iOutputQty; ++i)
//		cout << " " << setw(6) << sam.dYEval[i]; // by conventional serial method
//	cout << "\t~"; // feed forward evaluation
//	for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){
//		if(abs(sam.dYEval[i]-sam.dAltYEval[i])>0.0000005) cout << ">" << ++iReturn << ">";
//		cout << " " << setw(6) << sam.dAltYEval[i]; // by parallel GPU 
//	}
//	cout << "\n" << setprecision(6);
//	for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ //displaying complex-converted values for outputs
//		cout << S << "/" << i << "OTp " << setw(9) << sam.cdcDOutputs[i].x << "\t" << setw(9) << sam.cdcDOutputs[i].y << "i\t"; 
//			if((abs(sam.cdcAltDOutputs[i].x-sam.cdcDOutputs[i].x)+abs(sam.cdcAltDOutputs[i].y-sam.cdcDOutputs[i].y))>0.0000005) cout << ">" << ++iReturn << ">";
//		cout << "~ " << setw(9) << sam.cdcAltDOutputs[i].x << "\t" << setw(9) << sam.cdcAltDOutputs[i].y << "i\n"; 
//		cout << S << "/" << i << "Eval" << setw(9) << sam.cdcYEval[i].x << "\t" << setw(9) << sam.cdcYEval[i].y << "i\t"; 
//			if((abs(sam.cdcYEval[i].x-sam.cdcAltYEval[i].x)+abs(sam.cdcYEval[i].y-sam.cdcAltYEval[i].y))-0.0000005>0) cout << ">" << ++iReturn << ">";
//		cout << "~ " << setw(9) << sam.cdcAltYEval[i].x << "\t" << setw(9) << sam.cdcAltYEval[i].y << "i\n"; 
//	}
//	
//	cout << iReturn << " significant anomalies found\n";
//	if (bWait) // wait on keystroke or not
//		{ mExitKeystroke}
//	return iReturn;
//}


int ShowMeGPULS(struct rohanContext& rSes, long bWait)
{/// displays labeled contents of rLearn
	int iReturn=0;
	////iReturn=knlShowMeGPULS(rSes);
	if (bWait) // wait on keystroke or not
		{ mExitKeystroke}
	return iReturn;
}

//int ShowMeES(struct rohanContext& rSes, long bWait)
//{/// displays labeled contents of rLearn with evaluated outputs
//	int iReturn=0;
//	//cublasStatus csStatus;
//
//	printf("SHOWME Evaluations of %s\n", rSes.rLearn->sLearnSet); 
//	printf("Samples %d, Values/Line %d = Inputs %d + Outputs %d\n", rSes.rLearn->lSampleQty, rSes.rLearn->iValuesPerLine, rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
//	// 0 is ContActivation outputs, 1 is discrete outputs
//	if(rSes.rLearn->iEvalMode)
//		printf("Eval Mode %d Discrete outputs.\n", rSes.rLearn->iEvalMode);
//	else
//		printf("Eval Mode %d Continuous outputs.\n", rSes.rLearn->iEvalMode);
//	//commence with listing values
//	cout << fixed;
//	for(long s=0; s<rSes.rLearn->lSampleQty; ++s){
//		struct rohanSample& sam = rSes.rLearn->rSample[s];
//		cout << setw(4) << s << ":" << setprecision(2);
//		for (int i=0; i<=rSes.rLearn->iValuesPerLine; ++i){
//			cout << " " << setw(3) << sam.dXInputs[i];
//			if (i==rSes.rLearn->iInputQty)
//				cout << "\tO:"; // outputs begin here
//			if (i==(rSes.rLearn->iInputQty+rSes.rLearn->iOutputQty) && i<rSes.rLearn->iValuesPerLine)
//				cout << "\tX:"; // excess values begin here
//		}
//		cout << "\n" << setprecision(6);
//		cout << "   Y:" << setprecision(2); // feed forward evaluation
//		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i)
//			cout << " " << setw(6) << sam.dYEval[i]; // by conventional serial method
//		cout << "\t~"; // feed forward evaluation
//		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){
//			if(abs(sam.dAltYEval[i]-sam.dYEval[i])>0.0000005) cout << ">>";
//			cout << " " << setw(6) << sam.dAltYEval[i]; // by parallel GPU 
//		}
//		cout << "\n" << setprecision(6);
//	}
//	if (bWait) // wait on keystroke or not
//		{ mExitKeystroke}
//	return iReturn;
//}



//int TestPatWS(struct rohanContext& rSes, long bWait)
//{/// fills Weight Set and associated structures with known values, transfers to GPU mem and back to check for fidelity
//	cublasStatus csStatus;
//	int iReturn=0;
//	
//	for (int i=1; i < rSes.rNet->iLayerQty; ++i){ //no weights for layer 0
//		struct rohanLayer& lay = rSes.rNet->rLayer[i];
//		for (int j=0; j <= lay.iNeuronQty; ++j){ // no weights for neuron 0, but let's check anyway
//			for (int k=0; k <= lay.iDendriteQty; ++k){
//				cuDoubleComplex& way = lay.Weights[IDX2C(j,k, lay.iNeuronQty+1)];
//				way.x=0+(i*100.0 + j*10.0 + k);
//				way.y=0-(i*100.0 + j*10.0 + k);
//			}
//			int k=0;
//			lay.ZOutputs[j].x=0+(i*100.0 + j*10.0 + k);
//			lay.ZOutputs[j].y=0-(i*100.0 + j*10.0 + k);
//			lay.Deltas[j].x =0+(i*100.0 + j*10.0 + k);
//			lay.Deltas[j].y =0-(i*100.0 + j*10.0 + k);
//		}
//		csStatus=cublasSetMatrix(lay.iNeuronQty+1, lay.iDendriteQty+1, sizeof(cuDoubleComplex),
//			lay.Weights, lay.iNeuronQty+1, lay.Weights, lay.iNeuronQty+1);
//			mCuMsg(csStatus,"cublasSetMatrix()")
//		csStatus=cublasGetMatrix(lay.iNeuronQty+1, lay.iDendriteQty+1, sizeof(cuDoubleComplex),
//			lay.Weights, lay.iNeuronQty+1, lay.cdcAltWeights, lay.iNeuronQty+1);
//			mCuMsg(csStatus,"cublasGetMatrix()")
//		csStatus=cublasSetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.ZOutputs, 1, lay.ZOutputs, 1);
//			mCuMsg(csStatus,"cublasSetVector()")
//		csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.ZOutputs, 1, lay.cdcZAltOutputs, 1);
//			mCuMsg(csStatus,"cublasGetVector()")
//		csStatus=cublasSetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.Deltas, 1, lay.Deltas, 1);
//			mCuMsg(csStatus,"cublasSetVector()")
//		csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.Deltas, 1, lay.Deltas, 1);
//			mCuMsg(csStatus,"cublasGetVector()")
//	}
//	if (bWait) mExitKeystroke
//	return iReturn;
//}

int ShowMeArch(struct rohanContext& rSes, long bWait)
{/// displays network architectures and other parameters
	//cublasStatus csStatus;
	int iReturn=0;
	
	printf("\n== SHOWME Architectures ==\n");
	printf("iSectorQty %d\n", rSes.rNet->iSectorQty);
	printf("iLayerQty %d\n", rSes.rNet->iLayerQty);
	printf("iWeightMode %d\n", rSes.rNet->iWeightMode);
	printf("dK_DIV_TWO_PI %f\n", rSes.rNet->dK_DIV_TWO_PI);
	printf("two_pi_div_sect_qty %f\n", rSes.rNet->two_pi_div_sect_qty);
	printf("sWeightSet %s\n", rSes.rNet->sWeightSet);

	if(rSes.rNet->rLayer[1].iNeuronQty>5){

			for (int i=0; i < rSes.rNet->iLayerQty; ++i){
				struct rohanLayer& lay = rSes.rNet->rLayer[i];
				printf("Layer %d: %d dendrites, %d neurons\n", i, lay.iDendriteQty, lay.iNeuronQty );
			}
			//for (int i=0; i < rSes.rNet->iLayerQty; ++i){ //no weights for layer 0
			//	struct rohanLayer& lay = rSes.rNet->rLayer[i];
			//	if (i){
			//		csStatus=cublasGetMatrix(lay.iNeuronQty+1, lay.iDendriteQty+1, sizeof(cuDoubleComplex),
			//			lay.Weights, lay.iNeuronQty+1, lay.cdcAltWeights, lay.iNeuronQty+1);
			//			mCuMsg(csStatus,"cublasGetMatrix()")
			//	}
			//	csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.ZOutputs, 1, lay.cdcZAltOutputs, 1);
			//		mCuMsg(csStatus,"cublasGetVector()")
			//	csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.Deltas, 1, lay.Deltas, 1);
			//		mCuMsg(csStatus,"cublasGetVector()")
			//	cout << setiosflags(ios::right);
			//	cout << resetiosflags(ios::left);
			//	cout << setprecision(6) << fixed;
			//	for (int j=0; j <= lay.iNeuronQty; ++j){ // no weights for neuron 0, but let's check anyway
			//		if (i){
			//			for (int k=0; k <= lay.iDendriteQty; ++k){
			//				cuDoubleComplex& way = lay.Weights[IDX2C(j,k,lay.iNeuronQty+1)];
			//				cuDoubleComplex& tild = lay.cdcAltWeights[IDX2C(j,k,lay.iNeuronQty+1)];
			//				if((abs(tild.x-way.x)+abs(tild.y-way.y))>.0000005) {
			//					cout << i << "/" << j << "/" << k << "w " << setw(11) << way.x << "\t" << setw(11) << way.y << "i\t";
			//					cout << ">>";
			//					cout << "~ " << setw(11) << tild.x << "\t" << setw(11) << tild.y << "i\n";
			//				}
			//			}
			//		}
			//		else 
			//			cout << i << "/" << j << "/" << "-" << "w " << "\t" << "i\t" << "~ " << "\t" << "i\n";
			//		if((abs(lay.cdcZAltOutputs[j].x-lay.ZOutputs[j].x)+abs(lay.cdcZAltOutputs[j].y-lay.ZOutputs[j].y))>.0000005){
			//			cout << " Output" << setw(11) << lay.ZOutputs[j].x << "\t" << setw(11) << lay.ZOutputs[j].y << "i\t";
			//			cout << ">>";
			//			cout << "~ " << setw(11) << lay.cdcZAltOutputs[j].x << "\t" << setw(11) << lay.cdcZAltOutputs[j].y << "i\n"; 
			//		}
			//		if (i){
			//			if((abs(lay.Deltas[j].x-lay.Deltas[j].x)+abs(lay.Deltas[j].y-lay.Deltas[j].y))>.0000005){
			//				cout << " Delta " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\t"; 
			//				cout << ">>";
			//				cout << "~ " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\n"; 
			//			}
			//		}
			//	}
			//}
	}
	else {

			for (int i=0; i < rSes.rNet->iLayerQty; ++i){
				struct rohanLayer& lay = rSes.rNet->rLayer[i];
				cout << "Layer " << i << ": " << lay.iDendriteQty << " dendrites, " << lay.iNeuronQty << " neurons\n" ;
			}
			//for (int i=0; i < rSes.rNet->iLayerQty; ++i){ //no weights for layer 0
			//	struct rohanLayer& lay = rSes.rNet->rLayer[i];
			//	if (i){
			//		csStatus=cublasGetMatrix(lay.iNeuronQty+1, lay.iDendriteQty+1, sizeof(cuDoubleComplex),
			//			lay.Weights, lay.iNeuronQty+1, lay.cdcAltWeights, lay.iNeuronQty+1);
			//			mCuMsg(csStatus,"cublasGetMatrix()")
			//	}
			//	csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.ZOutputs, 1, lay.cdcZAltOutputs, 1);
			//		mCuMsg(csStatus,"cublasGetVector()")
			//	csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.Deltas, 1, lay.Deltas, 1);
			//		mCuMsg(csStatus,"cublasGetVector()")
			//	cout << setiosflags(ios::right);
			//	cout << resetiosflags(ios::left);
			//	cout << setprecision(6) << fixed;
			//	for (int j=0; j <= lay.iNeuronQty; ++j){ // no weights for neuron 0, but let's check anyway
			//		if (i){
			//			for (int k=0; k <= lay.iDendriteQty; ++k){
			//				cuDoubleComplex& way = lay.Weights[IDX2C(j,k,lay.iNeuronQty+1)];
			//				cuDoubleComplex& tild = lay.cdcAltWeights[IDX2C(j,k,lay.iNeuronQty+1)];
			//				cout << i << "/" << j << "/" << k << "w " << setw(11) << way.x << "\t" << setw(11) << way.y << "i\t";
			//					if((abs(tild.x-way.x)+abs(tild.y-way.y))>.0000005) cout << ">>";
			//				cout << "~ " << setw(11) << tild.x << "\t" << setw(11) << tild.y << "i\n";
			//			}
			//		}
			//		else 
			//			cout << i << "/" << j << "/" << "-" << "w " << "\t" << "i\t" << "~ " << "\t" << "i\n";
			//		cout << " Output" << setw(11) << lay.ZOutputs[j].x << "\t" << setw(11) << lay.ZOutputs[j].y << "i\t";
			//			if((abs(lay.cdcZAltOutputs[j].x-lay.ZOutputs[j].x)+abs(lay.cdcZAltOutputs[j].y-lay.ZOutputs[j].y))>.0000005) cout << ">>";
			//		cout << "~ " << setw(11) << lay.cdcZAltOutputs[j].x << "\t" << setw(11) << lay.cdcZAltOutputs[j].y << "i\n"; 
			//		if (i){
			//			cout << " Delta " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\t"; 
			//				if((abs(lay.Deltas[j].x-lay.Deltas[j].x)+abs(lay.Deltas[j].y-lay.Deltas[j].y))>.0000005) cout << ">>";
			//			cout << "~ " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\n"; 
			//		}
			//	}
			//}
	}
	if (bWait) mExitKeystroke
	return iReturn;
}
//
//int ShowMeWS(struct rohanContext& rSes, long bWait)
//{/// displays labeled parts of Weight Set and associated values
//	cublasStatus csStatus;
//	int iReturn=0;
//	
//	printf("SHOWME Weight Set: %s\n", rSes.rNet->sWeightSet);
//	printf("Layers %d, Inputs %d, Outputs %d\n", rSes.rNet->iLayerQty, rSes.rNet->rLayer[0].iNeuronQty, rSes.rNet->rLayer[rSes.rNet->iLayerQty-1].iNeuronQty);
//	cout << "Sectors: " << rSes.rNet->iSectorQty << " / " << rSes.rNet->dK_DIV_TWO_PI << " = " << rSes.rNet->iSectorQty/rSes.rNet->dK_DIV_TWO_PI << endl;
//	
//	if(rSes.rNet->rLayer[1].iNeuronQty>5){
//
//			for (int i=0; i < rSes.rNet->iLayerQty; ++i){
//				struct rohanLayer& lay = rSes.rNet->rLayer[i];
//				cout << "Layer " << i << ": " << lay.iDendriteQty << " dendrites, " << lay.iNeuronQty << " neurons\n" ;
//			}
//			for (int i=0; i < rSes.rNet->iLayerQty; ++i){ //no weights for layer 0
//				struct rohanLayer& lay = rSes.rNet->rLayer[i];
//				if (i){
//					csStatus=cublasGetMatrix(lay.iNeuronQty+1, lay.iDendriteQty+1, sizeof(cuDoubleComplex),
//						lay.Weights, lay.iNeuronQty+1, lay.cdcAltWeights, lay.iNeuronQty+1);
//						mCuMsg(csStatus,"cublasGetMatrix()")
//				}
//				csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.ZOutputs, 1, lay.cdcZAltOutputs, 1);
//					mCuMsg(csStatus,"cublasGetVector()")
//				csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.Deltas, 1, lay.Deltas, 1);
//					mCuMsg(csStatus,"cublasGetVector()")
//				cout << setiosflags(ios::right);
//				cout << resetiosflags(ios::left);
//				cout << setprecision(6) << fixed;
//				for (int j=0; j <= lay.iNeuronQty; ++j){ // no weights for neuron 0, but let's check anyway
//					if (i){
//						for (int k=0; k <= lay.iDendriteQty; ++k){
//							cuDoubleComplex& way = lay.Weights[IDX2C(j,k,lay.iNeuronQty+1)];
//							cuDoubleComplex& tild = lay.cdcAltWeights[IDX2C(j,k,lay.iNeuronQty+1)];
//							if((abs(tild.x-way.x)+abs(tild.y-way.y))>.0000005) {
//								cout << i << "/" << j << "/" << k << "w " << setw(11) << way.x << "\t" << setw(11) << way.y << "i\t";
//								cout << ">>";
//								cout << "~ " << setw(11) << tild.x << "\t" << setw(11) << tild.y << "i\n";
//							}
//						}
//					}
//					else 
//						cout << i << "/" << j << "/" << "-" << "w " << "\t" << "i\t" << "~ " << "\t" << "i\n";
//					if((abs(lay.cdcZAltOutputs[j].x-lay.ZOutputs[j].x)+abs(lay.cdcZAltOutputs[j].y-lay.ZOutputs[j].y))>.0000005){
//						cout << " Output" << setw(11) << lay.ZOutputs[j].x << "\t" << setw(11) << lay.ZOutputs[j].y << "i\t";
//						cout << ">>";
//						cout << "~ " << setw(11) << lay.cdcZAltOutputs[j].x << "\t" << setw(11) << lay.cdcZAltOutputs[j].y << "i\n"; 
//					}
//					if (i){
//						if((abs(lay.Deltas[j].x-lay.Deltas[j].x)+abs(lay.Deltas[j].y-lay.Deltas[j].y))>.0000005){
//							cout << " Delta " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\t"; 
//							cout << ">>";
//							cout << "~ " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\n"; 
//						}
//					}
//				}
//			}
//	}
//	else {
//
//			for (int i=0; i < rSes.rNet->iLayerQty; ++i){
//				struct rohanLayer& lay = rSes.rNet->rLayer[i];
//				cout << "Layer " << i << ": " << lay.iDendriteQty << " dendrites, " << lay.iNeuronQty << " neurons\n" ;
//			}
//			for (int i=0; i < rSes.rNet->iLayerQty; ++i){ //no weights for layer 0
//				struct rohanLayer& lay = rSes.rNet->rLayer[i];
//				if (i){
//					csStatus=cublasGetMatrix(lay.iNeuronQty+1, lay.iDendriteQty+1, sizeof(cuDoubleComplex),
//						lay.Weights, lay.iNeuronQty+1, lay.cdcAltWeights, lay.iNeuronQty+1);
//						mCuMsg(csStatus,"cublasGetMatrix()")
//				}
//				csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.ZOutputs, 1, lay.cdcZAltOutputs, 1);
//					mCuMsg(csStatus,"cublasGetVector()")
//				csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.Deltas, 1, lay.Deltas, 1);
//					mCuMsg(csStatus,"cublasGetVector()")
//				cout << setiosflags(ios::right);
//				cout << resetiosflags(ios::left);
//				cout << setprecision(6) << fixed;
//				for (int j=0; j <= lay.iNeuronQty; ++j){ // no weights for neuron 0, but let's check anyway
//					if (i){
//						for (int k=0; k <= lay.iDendriteQty; ++k){
//							cuDoubleComplex& way = lay.Weights[IDX2C(j,k,lay.iNeuronQty+1)];
//							cuDoubleComplex& tild = lay.cdcAltWeights[IDX2C(j,k,lay.iNeuronQty+1)];
//							cout << i << "/" << j << "/" << k << "w " << setw(11) << way.x << "\t" << setw(11) << way.y << "i\t";
//								if((abs(tild.x-way.x)+abs(tild.y-way.y))>.0000005) cout << ">>";
//							cout << "~ " << setw(11) << tild.x << "\t" << setw(11) << tild.y << "i\n";
//						}
//					}
//					else 
//						cout << i << "/" << j << "/" << "-" << "w " << "\t" << "i\t" << "~ " << "\t" << "i\n";
//					cout << " Output" << setw(11) << lay.ZOutputs[j].x << "\t" << setw(11) << lay.ZOutputs[j].y << "i\t";
//						if((abs(lay.cdcZAltOutputs[j].x-lay.ZOutputs[j].x)+abs(lay.cdcZAltOutputs[j].y-lay.ZOutputs[j].y))>.0000005) cout << ">>";
//					cout << "~ " << setw(11) << lay.cdcZAltOutputs[j].x << "\t" << setw(11) << lay.cdcZAltOutputs[j].y << "i\n"; 
//					if (i){
//						cout << " Delta " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\t"; 
//							if((abs(lay.Deltas[j].x-lay.Deltas[j].x)+abs(lay.Deltas[j].y-lay.Deltas[j].y))>.0000005) cout << ">>";
//						cout << "~ " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\n"; 
//					}
//				}
//			}
//
//
//	}
//	if (bWait) mExitKeystroke
//	return iReturn;
//}
//
//int ShowMeLayer(struct rohanContext& rSes, int L, long bWait)
//{/// displays labeled parts of Weight Set and associated values
//	cublasStatus csStatus;
//	int iReturn=0;
//	
//	printf("SHOWME Layer %d\n",L);
//	int i = abs(L); //if L is negative, show only divergent weights
//	struct rohanLayer& lay = rSes.rNet->rLayer[i];
//	struct rohanLayer& olay = rSes.rNet->rLayer[i-1];
//	if (i){
//		csStatus=cublasGetMatrix(lay.iNeuronQty+1, lay.iDendriteQty+1, sizeof(cuDoubleComplex),
//			lay.Weights, lay.iNeuronQty+1, lay.cdcAltWeights, lay.iNeuronQty+1);
//			mCuMsg(csStatus,"cublasGetMatrix()")
//	}
//	csStatus=cublasGetVector(olay.iNeuronQty+1, sizeof(cuDoubleComplex), olay.ZOutputs, 1, olay.cdcZAltOutputs, 1);
//		mCuMsg(csStatus,"cublasGetVector()")
//	csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.ZOutputs, 1, lay.cdcZAltOutputs, 1);
//		mCuMsg(csStatus,"cublasGetVector()")
//	csStatus=cublasGetVector(lay.iNeuronQty+1, sizeof(cuDoubleComplex), lay.Deltas, 1, lay.Deltas, 1);
//		mCuMsg(csStatus,"cublasGetVector()")
//	cout << setiosflags(ios::right);
//	cout << resetiosflags(ios::left);
//	cout << setprecision(6) << fixed;
//	for (int j=0; j <= olay.iNeuronQty; ++j){ // no weights for neuron 0, but let's check anyway
//		if(L<0){
//			if((abs(olay.cdcZAltOutputs[j].x-olay.ZOutputs[j].x)+abs(olay.cdcZAltOutputs[j].y-olay.ZOutputs[j].y))>.0000005){
//				cout << "  Input" << setw(11) << olay.ZOutputs[j].x << "\t" << setw(11) << olay.ZOutputs[j].y << "i\t";
//				cout << ">>";
//				cout << "~ " << setw(11) << olay.cdcZAltOutputs[j].x << "\t" << setw(11) << olay.cdcZAltOutputs[j].y << "i\n"; 
//			}
//		}
//		else{
//			cout << "  Input" << setw(11) << olay.ZOutputs[j].x << "\t" << setw(11) << olay.ZOutputs[j].y << "i\t";
//			if((abs(olay.cdcZAltOutputs[j].x-olay.ZOutputs[j].x)+abs(olay.cdcZAltOutputs[j].y-olay.ZOutputs[j].y))>.0000005) cout << ">>";
//			cout << "~ " << setw(11) << olay.cdcZAltOutputs[j].x << "\t" << setw(11) << olay.cdcZAltOutputs[j].y << "i\n"; 
//		}
//	}
//	for (int j=0; j <= lay.iNeuronQty; ++j){ // no weights for neuron 0, but let's check anyway
//		if (i){
//			for (int k=0; k <= lay.iDendriteQty; ++k){
//				cuDoubleComplex& way = lay.Weights[IDX2C(j,k,lay.iNeuronQty+1)];
//				cuDoubleComplex& tild = lay.cdcAltWeights[IDX2C(j,k,lay.iNeuronQty+1)];
//				if(L<0){
//					if((abs(way.x-tild.x)+abs(way.y-tild.y))>.0000005) {
//						cout << i << "/" << j << "/" << k << "w " << setw(11) << way.x << "\t" << setw(11) << way.y << "i\t";
//						cout << ">>";
//						cout << "~ " << setw(11) << tild.x << "\t" << setw(11) << tild.y << "i\n";
//					}
//				}
//				else{
//					cout << i << "/" << j << "/" << k << "w " << setw(11) << way.x << "\t" << setw(11) << way.y << "i\t";
//					if((abs(way.x-tild.x)+abs(way.y-tild.y))>.0000005) cout << ">>";
//					cout << "~ " << setw(11) << tild.x << "\t" << setw(11) << tild.y << "i\n";
//				}
//			}
//		}
//		else
//			cout << i << "/" << j << "/" << "-" << "w " << "\t" << "i\t" << "~ " << "\t" << "i\n";
//		if(L<0){
//			if((abs(lay.cdcZAltOutputs[j].x-lay.ZOutputs[j].x)+abs(lay.cdcZAltOutputs[j].y-lay.ZOutputs[j].y))>.0000005) {
//				cout << " Output" << setw(11) << lay.ZOutputs[j].x << "\t" << setw(11) << lay.ZOutputs[j].y << "i\t";
//				cout << ">>";
//				cout << "~ " << setw(11) << lay.cdcZAltOutputs[j].x << "\t" << setw(11) << lay.cdcZAltOutputs[j].y << "i\n"; 
//			}
//		}
//		else{
//			cout << " Output" << setw(11) << lay.ZOutputs[j].x << "\t" << setw(11) << lay.ZOutputs[j].y << "i\t";
//			if((abs(lay.cdcZAltOutputs[j].x-lay.ZOutputs[j].x)+abs(lay.cdcZAltOutputs[j].y-lay.ZOutputs[j].y))>.0000005) cout << ">>";
//			cout << "~ " << setw(11) << lay.cdcZAltOutputs[j].x << "\t" << setw(11) << lay.cdcZAltOutputs[j].y << "i\n"; 
//		}
//		if (i){
//			if(L<0){
//				if((abs(lay.Deltas[j].x-lay.Deltas[j].x)+abs(lay.Deltas[j].y-lay.Deltas[j].y))>.0000005) {
//					cout << " Delta " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\t"; 
//					cout << ">>";
//					cout << "~ " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\n"; 
//				}
//			}
//			else{
//				cout << " Delta " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\t"; 
//				if((abs(lay.Deltas[j].x-lay.Deltas[j].x)+abs(lay.Deltas[j].y-lay.Deltas[j].y))>.0000005) cout << ">>";
//				cout << "~ " << setw(11) << lay.Deltas[j].x << "\t" << setw(11) << lay.Deltas[j].y << "i\n"; 
//			}
//		}		
//	}
//	
//	if (bWait) mExitKeystroke
//	return iReturn;
//}
//
//int ShowMeGPUOS(struct rohanContext& rSes, long bWait)
//{/// displays outputs of GPU calculations
//	int iReturn=0;
//	iReturn=knlShowMeGPUnnOut(rSes);
//	if (bWait) // wait on keystroke or not
//		{ mExitKeystroke}
//	return iReturn;
//}
//

//int ShowMeErr(struct rohanContext& rSes, long bWait)
//{/// displays labeled parts of session context and associated values, plus evaluation errors
//	int iReturn=0;
//
//	printf("SHOWME Errors\n");
//	printf("Samples requested: %d\n", rSes.rLearn->lSampleQty); // size of working subset of samples
//	printf("Target RMSE: %f\n", rSes.dTargetRMSE); // target RMSE
//	printf("MAX threshold %f\n", rSes.dMAX); // dont backprop unless error is this much
//	printf("Trainable Samples %d\n", rSes.lSamplesTrainable);
//	if (rSes.bContActivation) printf("Continuous Activation\n"); // use Continuous activation function
//	else printf("Discrete Activation\n");
//	printf("Last RMSE: %f\n", rSes.dRMSE); // evaluated RMSE
//	
//	for (long s=0; s<rSes.rLearn->lSampleQty; ++s){
//		struct rohanSample& sam=rSes.rLearn->rSample[s];
//		double errCont=abs(sam.dDOutputs[1]-sam.dYEval[1]);
//		double errDisc=abs(sam.dDOutputs[1]-floor(sam.dYEval[1]));
//		cout << s << ":\t" << sam.dDOutputs[1] << "\t~" << sam.dYEval[1] << "\tc" << errCont << "\td" << errDisc << "\t:" << ++iReturn << "\n";
//	}
//	if (bWait) mExitKeystroke
//	return iReturn;
//}

