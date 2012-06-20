/* Includes, cuda */
#include "stdafx.h"
#include "cuPrintf.cu"

extern int gDebugLvl, gDevDebug, gTrace;
extern float gElapsedTime, gKernelTimeTally;

// device-global variables to facilitate data transfer
__device__ __align__(16) __constant__ struct rohanContext devSes;
__device__ __align__(16) __constant__ struct rohanLearningSet devLearn;
__device__ __align__(16) struct rohanNetwork devNet;
__device__ __align__(16) const cuDoubleComplex gpuZero = { 0, 0 };
__device__ __align__(16) double devdReturn[1024*1024];
__device__ __align__(16) double devdRMSE=0;
__device__ __align__(16) int devlReturn[1024*1024];
__device__ __align__(16) int devlTrainable=0;
__device__ __align__(16) int gDevDebug=0;


extern"C"
int SetDevDebug(int gDevDebug)
{mIDfunc /*! controls debug flag in device memory */

	cudaMemcpyToSymbol( "gDevDebug", &gDevDebug, sizeof(int) );
		mCheckCudaWorked

	return gDevDebug;
}	


extern"C"
int knlBackProp(struct rohanContext& rSes, long lSampleQtyReq, long o, char Option, int iBlocks, int iThreads)
{mIDfunc /*! divides error in yielded values and back-propagates corrections among weights */
// Option S - single sample correction only
// Option E - keep existing weights, count trainable samples only
// Option R - perform corrections for all trainable samples; may assume FFE has just happened/classify based on pre-iteration weights
// Option I - perform corrections for all trainable samples; may assume FFE has just happened/classify based on intra-iteration weights
//
// multithreaded evaluation available

	cudaEvent_t start, stop;
	int lTotal=0; // ltotal declared
	
		cudaEventCreate( &start);
		cudaEventCreate( &stop);
	cudaPrintfInit();

		cudaEventRecord( start, 0);
	mtkBackPropMT<<< iBlocks , iThreads >>>( lSampleQtyReq, o, Option);
		cudaEventRecord( stop, 0);
		mCheckCudaWorked
	
	cudaMemcpyFromSymbol( &lTotal, "devlTrainable", sizeof(long) ); // retrieve return value
		mCheckCudaWorked
		cudaEventSynchronize( stop);
		cudaEventElapsedTime( &gElapsedTime, start, stop);
		gKernelTimeTally+=gElapsedTime;

	cudaPrintfDisplay(rSes.deviceBucket, true);
	cudaPrintfEnd();
		cudaEventDestroy( start);
		cudaEventDestroy( stop);

	return lTotal;
}


__global__ void mtkBackPropMT( long lSampleQtyReq, long o, char Option)
{/*! divides error in yielded values and back-propagates corrections among weights */
// Option S - single sample correction only
// Option E - keep existing weights, count trainable samples only
// Option R - perform corrections for all trainable samples; may assume FFE has just happened/classify based on pre-iteration weights
// Option I - perform corrections for all trainable samples; may assume FFE has just happened/classify based on intra-iteration weights YY
	
	if(threadIdx.x==0){
		cuPrintf(">>DEVICE: mtkBackPropMT( long %d, long %d, char %c);\n", lSampleQtyReq, o, Option);
		cuPrintf(">>DEVICE: mtkTestProp Wall= %08lX\n", subkCrc32buf((char*)(devNet.Wt), MAXWEIGHTS * 16) );
	}
	devlTrainable=0; // reset global mem trainable counter

	if(Option=='R' || Option=='r'){ //
		subkBackPropRoptMT(lSampleQtyReq, o);
	}
	else if(Option=='E' || Option=='e'){ //
		subkBackPropEoptMT(lSampleQtyReq, o);
	}
	else if(Option=='S' || Option=='s'){
		subkBackPropSoptMThread(lSampleQtyReq, false,  devNet, devNet.Signals, devNet.Zs, devNet.Wt, devNet.Deltas, devLearn.gpuXInputs, devLearn.gpuYEval, devLearn.gpudYEval);
	}
}


__device__ void subkBackPropRoptMT(long lSampleQtyReq, long o)
{/*! perform corrections for all trainable samples; may assume FFE has just happened/classify based on pre-iteration weights  */
	//
	// externalities: const devSes, const devLearn, devNet

	__shared__  __align__(16) struct rohanNetwork myDevNet; //set up copy in highest speed shared memory
	long index; // for warpwise loops
	long OUTROWLEN=devLearn.iOutputQty+1; // prepare array index and width
	long tIx = threadIdx.x + blockDim.x * blockIdx.x; // tIx is thread index over the kernel
	long lTotalThreads = gridDim.x * blockDim.x; // total number of threads
	double maxSquared = devSes.dMAX * devSes.dMAX ; //needed to compart to stored delta squared values
	
	myDevNet=devNet; // copy contents to shared memory for speed
	for (long s=0; s<lSampleQtyReq; ++s){ // iterate over samples
		//  It is assumed that FFE has just taken place prior to this call
		if( devLearn.gpudSqrErr[IDX2C( o, s, OUTROWLEN )] > maxSquared ){ // if the MAX criterion is exceeded	
			if(tIx==0){
				++devlTrainable; // increment the counter				
			}
			// do backprop
			subkBackPropSoptMThread( s, true, myDevNet, myDevNet.Signals, myDevNet.Zs, myDevNet.Wt, myDevNet.Deltas, devLearn.gpuXInputs, devLearn.gpuYEval, devLearn.gpudYEval);
		}
	}
	// weight adjustment loop ends, return weights to global mem for evaluate kernel
	for (long offset=0; (index =offset+tIx)< MAXWEIGHTS ; offset+=lTotalThreads){ // iterate over weights
		devNet.Wt[index]=myDevNet.Wt[index]; // copy to global back from shared memory
	}
}	// exit to allow eval kernel


__device__ void subkBackPropSoptMThread(long s, int o, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{/*! propagates adjustment of weights backwards preceeding layers from the chosen network output. */
	// s is sample's index
	//
	// works with 32, 64, 128, 256, or 512 threads on 1 block  5/24/12 - problem with RMSE 5/26/12
	// externalities: devSes, devLearn
	//
	{
		// proceed with single-warp, multithread backprop procedure
		long index, kindex; // for warpwise loops
		long tIx = threadIdx.x + blockDim.x * blockIdx.x; // tIx is thread index over the kernel
		long lTotalThreads = gridDim.x * blockDim.x; // total number of threads
		/* clear all temp values BP0 */
		__shared__  __align__(16) cuDoubleComplex Signals[MAXNEURONS], Zs[MAXNEURONS], Deltas[MAXNEURONS];
		for (long offset=0;	(index =offset+threadIdx.x)< MAXNEURONS; offset+=blockDim.x){ 
			Deltas[index]=gpuZero;
			Signals[index]=gpuZero;
			Zs[index]=gpuZero; // preloading weights reduces mass R RMSE/eval time from 3.0 ms to 2.6 ms, so we're going try it here as well: JAW 5/27/12
		}
		/* re-evaluate sample to load temp values. BPI */
		// this sample eval includes __syncthreads() internally
		subkEvalSampleBetaMT( devSes, s, Net, o, Signals, Zs, Wt, XInputs, YEval, dYEval, devLearn.gpudSqrErr);
		/* begin error calculation. BPII */
		// Deltastar /* measured error at the chosen network output. */ ;
		/* calc top layer deltas. */
		long TOP=Net.iLayerQty-1;
		int ROWLEN=Net.iNeuronQTY[TOP];
		//for(int i=0; i<Net.iNeuronQTY[TOP]; ++i){
		for (long offset=0; (index =offset+tIx)< Net.iNeuronQTY[TOP] ; offset+=lTotalThreads){ // index stands for i
			 // delta-star = D - Y = Desired output minus actual output from evaluation
			 // D is the cplx coords of the sector of the desired answer		Y is the complex result of evaluation of the given sample, unactivated. */
			cuDoubleComplex Deltastar = CxSubtractCxUT( 
							devLearn.gpuDOutputs[ IDX2C( index, s, ROWLEN ) ], 
							Signals[Net.iNeuronOfst[TOP]+index] );
			 /* divide the correction; delta = alpha * delta-star / n+1 (but alpha is always 1 for now). */
			//Deltas[Net.iNeuronOfst[TOP]+index] = CxDivideRlUT( Deltastar, Net.iDendrtQTY[TOP] );
			Deltas[Net.iNeuronOfst[TOP]+index] = CxMultiplyRlUT( Deltastar, Net.dINV_S[TOP] );
		}
		/* Now distribute the correction to lower layers if any. BPII.1 */
		if (Net.iLayerQty>2){  /* remember layer 0 = inputs, layer 1 = bottom row, layer {2..iLayerQty-2} = middle row, layer iLayerQty-1 = top row. */
			for (int L=Net.iLayerQty-1; L>1; --L){
				long LAY = L; /* setup access to layers. */
				long TRIB = L-1; /* trib for tributary.*/
				int iTributQTY=Net.iNeuronQTY[TRIB];
				//int Sj=Net.iDendrtQTY[TRIB]; if (TRIB==1) Sj=1; // Sj=1 for firest hidden layer
				for (int i=1; i<Net.iNeuronQTY[LAY]; ++i) { // skip 0th neuron as its weights are either 1 (div identity) or 0 (div forbidden) and don't change anyway
					// k index must begin at 1, neuron zero not valid for correction
					//for (int k=1; k<iTributQTY; ++k) { /* the contribution to ith neuron's kth tributary's delta = i's delta/i's weight k. */
					for (long offset=1; ( kindex =offset+tIx)< iTributQTY ; offset+=lTotalThreads){ // kindex stands for k
									  Deltas[Net.iNeuronOfst[TRIB]+kindex] 
						= CxAddCxUT ( Deltas[Net.iNeuronOfst[TRIB]+kindex] , 
							CxDivideCxUT( 
								Deltas[Net.iNeuronOfst[LAY]+i] , 
								Wt[IDX2C( Net.iWeightOfst[LAY]+kindex, i, iTributQTY )] ));
					}
					//__syncthreads(); YY
				}
				// k index must begin at 1, neuron zero not valid for correction
				//for (int k=1; k<iTributQTY; ++k) { /* contributions accumulated, now divide by dendrites+1. */
				for (long offset=1; ( kindex =offset+tIx)< iTributQTY ; offset+=lTotalThreads){ // kindex stands for k
					//cuDoubleComplex preDiv=Deltas[Net.iNeuronOfst[TRIB]+kindex]; // diagnostic purpose only, remove if removing other diags
					//Deltas[Net.iNeuronOfst[TRIB]+kindex] 
					//	= CxDivideRlUT( 
					//		Deltas[Net.iNeuronOfst[TRIB]+kindex] , 
					//		Sj );
					Deltas[Net.iNeuronOfst[TRIB]+kindex] 
						= CxMultiplyRlUT( 
							Deltas[Net.iNeuronOfst[TRIB]+kindex] , 
							Net.dINV_S[TRIB] );
				}
			}
		}
		/* error distribution completed */
		/* and now update the weights BP III */
		/* adj weights on first hidden layer. */
			int FHID = 1;
			int SIG = 0;
			int iSignalQTY=Net.iNeuronQTY[SIG]; //rSes.rLearn->iInputQty+1;
			int iHidWidth=Net.iNeuronQTY[FHID];
		for (int k=1; k<iHidWidth; ++k){
			//for (int i=0; i<iSignalQTY; ++i){  
			for (long offset=0; ( index =offset+tIx)< iSignalQTY ; offset+=lTotalThreads){ // index stands for i
				/* dW=d*xbar/s1/|z|= neuron's delta * input's conjugate / ( dendrites+1 * abs of input i ). */
							Wt[IDX2C( Net.iWeightOfst[FHID]+index, k, iSignalQTY )]
				=CxAddCxUT( Wt[IDX2C( Net.iWeightOfst[FHID]+index, k, iSignalQTY )] , 
					CxDivideRlUT( 
						CxMultiplyCxUT( 
							Deltas[Net.iNeuronOfst[FHID]+k] , 
							CxConjugateUT( Signals[Net.iNeuronOfst[SIG]+index] ) 
						) , 
						CxAbsUT( Zs[Net.iNeuronOfst[FHID]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
					)
				);
			}
		}
		/* re-evaluate sample to update temp values. */
		subkEvalSampleBetaMT( devSes, s, Net, false, Signals, Zs, Wt, XInputs, YEval, dYEval, devLearn.gpudSqrErr);
		if (Net.iLayerQty>2){
			 /* now use those outputs' conjugates and the deltas to adjust middle layers. BP III.1 */
			for (int L=2; L<Net.iLayerQty-1; ++L){
				 /* setup access to layers. */
				long LAY = L;
				long TRIB = L-1;
				//int iLayWidth=Net.iNeuronQTY[LAY];
				int iTribWidth=Net.iNeuronQTY[TRIB];
				for (int k=1; k<Net.iNeuronQTY[LAY]; ++k){
					//for (int i=0; i<Net.iNeuronQTY[TRIB]; ++i){  
					for (long offset=0; ( index =offset+tIx)< Net.iNeuronQTY[TRIB] ; offset+=lTotalThreads){ // index stands for i
						/* the adjustment added to kth neuron's ith trib's weight = k's delta * complex conjugate of i's signal / (abs of k's previous-wt product-sum * dendrites+1)  . */
									Wt[IDX2C( Net.iWeightOfst[LAY]+index, k, iTribWidth )]
						=CxAddCxUT( Wt[IDX2C( Net.iWeightOfst[LAY]+index, k, iTribWidth )] , 
							CxDivideRlUT( 
								CxMultiplyCxUT( 
									Deltas[Net.iNeuronOfst[LAY]+k] , 
									CxConjugateUT( Signals[Net.iNeuronOfst[TRIB]+index] ) 
								) ,
								( 
									CxAbsUT( Zs[Net.iNeuronOfst[LAY]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
								)
							)
						);
					}
				}
				/* layer is complete. */
				subkEvalSampleBetaMT( devSes, s, Net, true, Signals, Zs, Wt, XInputs, YEval, dYEval, devLearn.gpudSqrErr);
			}
		}
		/* correct output layer BP III.3 */
		long SUB = TOP-1; 
		//int iTopWidth=Net.iNeuronQTY[TOP];
		int iSubWidth=Net.iNeuronQTY[SUB];
		for (int k=1; k<Net.iNeuronQTY[TOP]; ++k){
			//for (int i=0; i<Net.iNeuronQTY[SUB]; ++i){  
			for (long offset=0; ( index =offset+tIx)< Net.iNeuronQTY[SUB] ; offset+=lTotalThreads){ // index stands for i
				/* For last layer only, adjustment to kth neuron's ith weight = k's delta * complex conjugate of i's signal / ( dendrites+1)  . */
							Wt[IDX2C( Net.iWeightOfst[TOP]+index, k, iSubWidth )]
				=CxAddCxUT( Wt[IDX2C( Net.iWeightOfst[TOP]+index, k, iSubWidth )] , 
					CxMultiplyCxUT( 
						Deltas[Net.iNeuronOfst[TOP]+k] , 
						CxConjugateUT( Signals[Net.iNeuronOfst[SUB]+index] ) 
					)
				);  // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
			}
		}
		/* backprop is complete. */
	//	/* re-evaluate sample to get new yields. BPIV */
	//	subkEvalSampleBetaMT( devSes, s, Net, o, Signals, Zs, Wt, XInputs, YEval, dYEval, devLearn.gpudSqrErr);
	//	 now, re-calculate delta-star for each output and ensure that value is zero (at least when alpha is 1) BPV
	//	for(int i=0; i<Net.iNeuronQTY[TOP]; ++i){
	//	for (long offset=0; ( index =offset+tIx)< Net.iNeuronQTY[TOP] ; offset+=lTotalThreads){ // index stands for i
	//		  delta-star = D - Y = Desired output minus actual output from evaluation
	//		  D is the cplx coords of the sector of the desired answer		Y is the complex result of evaluation of the given sample. */
	//		Deltastar = CxSubtractCxUT( devLearn.gpuDOutputs[ IDX2C( index, s, ROWLEN ) ], 
	//			Net.gpuSectorBdry[(int) dYEval[ IDX2C( index, s, ROWLEN ) ] ] );
	//		 delta-star = D - Y = Desired output minus actual output from evaluation
	//		double D =  devLearn.gpudDOutputs[ IDX2C( index, s, ROWLEN ) ];
	//		double Y = dYEval[ IDX2C( index, s, ROWLEN ) ];
	//	return iReturn; /* number of weights updated. */
	}
	// execution returns from Multi-Warp version
}

__device__ void subkBackPropSoptMWarp(long s, int o, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{/*! propagates adjustment of weights backwards preceeding layers from the chosen network output. */
	// This subroutine employs extensive use of __syncthreads() and other mechanisms to keep multiple warps from colliding or outrunning correct values
	// s is sample's index
	// 
	// working with 1 block and 64, 128, 256, or 512 threads 5/24/12

	long index, kindex; // for warpwise loops
	long tIx = threadIdx.x + blockDim.x * blockIdx.x; // tIx is thread index over the kernel
	long lTotalThreads = gridDim.x * blockDim.x; // total number of threads
	/* clear all temp values BP0 */
	for (long offset=0; (index =offset+tIx)< MAXNEURONS ; offset+=lTotalThreads){ // index stands for i
		Deltas[index]=gpuZero;
		Signals[index]=gpuZero;
		Zs[index]=gpuZero;
	}
	/* re-evaluate sample to load temp values. BPI */
	subkEvalSampleBetaMT( devSes, s, Net, (s==0), Signals, Zs, Wt, XInputs, YEval, dYEval, devLearn.gpudSqrErr);
	/* begin error calculation. BPII */
	cuDoubleComplex Deltastar /* measured error at the chosen network output. */ ;
	/* calc top layer deltas. */
	long TOP=Net.iLayerQty-1;
	int ROWLEN=Net.iNeuronQTY[TOP];
	//for(int i=0; i<Net.iNeuronQTY[TOP]; ++i){
	for (long offset=0; (index =offset+tIx)< Net.iNeuronQTY[TOP] ; offset+=lTotalThreads){ // index stands for i
		 // delta-star = D - Y = Desired output minus actual output from evaluation
		 // D is the cplx coords of the sector of the desired answer		Y is the complex result of evaluation of the given sample, unactivated. */
		Deltastar = CxSubtractCxUT( 
						devLearn.gpuDOutputs[ IDX2C( index, s, ROWLEN ) ], 
						Signals[Net.iNeuronOfst[TOP]+index] );
		 /* divide the correction; delta = alpha * delta-star / n+1 (but alpha is always 1 for now). */
		//Deltas[Net.iNeuronOfst[TOP]+index] = CxDivideRlUT( Deltastar, Net.iDendrtQTY[TOP] );
		Deltas[Net.iNeuronOfst[TOP]+index] = CxMultiplyRlUT( Deltastar, Net.dINV_S[TOP] );
	}
	/* Now distribute the correction to lower layers if any. BPII.1 */
	if (Net.iLayerQty>2){  /* remember layer 0 = inputs, layer 1 = bottom row, layer {2..iLayerQty-2} = middle row, layer iLayerQty-1 = top row. */
		for (int L=Net.iLayerQty-1; L>1; --L){
			long LAY = L; /* setup access to layers. */
			long TRIB = L-1; /* trib for tributary.*/
			int iTributQTY=Net.iNeuronQTY[TRIB];
			//int Sj=Net.iDendrtQTY[TRIB]; if (TRIB==1) Sj=1; // Sj=1 for firest hidden layer
			for (int i=1; i<Net.iNeuronQTY[LAY]; ++i) { // skip 0th neuron as its weights are either 1 (div identity) or 0 (div forbidden) and don't change anyway
				// k index must begin at 1, neuron zero not valid for correction
				//for (int k=1; k<iTributQTY; ++k) { /* the contribution to ith neuron's kth tributary's delta = i's delta/i's weight k. */
				for (long offset=1; ( kindex =offset+tIx)< iTributQTY ; offset+=lTotalThreads){ // kindex stands for k
								  Deltas[Net.iNeuronOfst[TRIB]+kindex] 
					= CxAddCxUT ( Deltas[Net.iNeuronOfst[TRIB]+kindex] , 
						CxDivideCxUT( 
							Deltas[Net.iNeuronOfst[LAY]+i] , 
							Wt[IDX2C( Net.iWeightOfst[LAY]+kindex, i, iTributQTY )] ));
				}
			}
			// k index must begin at 1, neuron zero not valid for correction
			//for (int k=1; k<iTributQTY; ++k) { /* contributions accumulated, now divide by dendrites+1. */
			for (long offset=1; ( kindex =offset+tIx)< iTributQTY ; offset+=lTotalThreads){ // kindex stands for k
				//cuDoubleComplex preDiv=Deltas[Net.iNeuronOfst[TRIB]+kindex]; // diagnostic purpose only, remove if removing other diags
				//Deltas[Net.iNeuronOfst[TRIB]+kindex] 
				//	= CxDivideRlUT( 
				//		Deltas[Net.iNeuronOfst[TRIB]+kindex] , 
				//		Sj );
				Deltas[Net.iNeuronOfst[TRIB]+kindex] 
					= CxMultiplyRlUT( 
						Deltas[Net.iNeuronOfst[TRIB]+kindex] , 
						Net.dINV_S[TRIB] );
			}
		}
	}
	/* error distribution completed */
	/* and now update the weights BP III */
	/* adj weights on first hidden layer. */
		int FHID = 1;
		int SIG = 0;
		int iSignalQTY=Net.iNeuronQTY[SIG]; //rSes.rLearn->iInputQty+1;
		int iHidWidth=Net.iNeuronQTY[FHID];
	for (int k=1; k<iHidWidth; ++k){
		//for (int i=0; i<iSignalQTY; ++i){  
		for (long offset=0; ( index =offset+tIx)< iSignalQTY ; offset+=lTotalThreads){ // index stands for i
			/* dW=d*xbar/s1/|z|= neuron's delta * input's conjugate / ( dendrites+1 * abs of input i ). */
						Wt[IDX2C( Net.iWeightOfst[FHID]+index, k, iSignalQTY )]
			=CxAddCxUT( Wt[IDX2C( Net.iWeightOfst[FHID]+index, k, iSignalQTY )] , 
				CxDivideRlUT( 
					CxMultiplyCxUT( 
						Deltas[Net.iNeuronOfst[FHID]+k] , 
						CxConjugateUT( Signals[Net.iNeuronOfst[SIG]+index] ) 
					) , 
					CxAbsUT( Zs[Net.iNeuronOfst[FHID]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
				)
			);
		}
	}
	/* re-evaluate sample to update temp values. */
	subkEvalSampleBetaMT( devSes, s, Net, false, Signals, Zs, Wt, XInputs, YEval, dYEval, devLearn.gpudSqrErr);
	if (Net.iLayerQty>2){
		 /* now use those outputs' conjugates and the deltas to adjust middle layers. BP III.1 */
		for (int L=2; L<Net.iLayerQty-1; ++L){
			 /* setup access to layers. */
			long LAY = L;
			long TRIB = L-1;
			//int iLayWidth=Net.iNeuronQTY[LAY];
			int iTribWidth=Net.iNeuronQTY[TRIB];
			for (int k=1; k<Net.iNeuronQTY[LAY]; ++k){
				//for (int i=0; i<Net.iNeuronQTY[TRIB]; ++i){  
				for (long offset=0; ( index =offset+tIx)< Net.iNeuronQTY[TRIB] ; offset+=lTotalThreads){ // index stands for i
					/* the adjustment added to kth neuron's ith trib's weight = k's delta * complex conjugate of i's signal / (abs of k's previous-wt product-sum * dendrites+1)  . */
								Wt[IDX2C( Net.iWeightOfst[LAY]+index, k, iTribWidth )]
					=CxAddCxUT( Wt[IDX2C( Net.iWeightOfst[LAY]+index, k, iTribWidth )] , 
						CxDivideRlUT( 
							CxMultiplyCxUT( 
								Deltas[Net.iNeuronOfst[LAY]+k] , 
								CxConjugateUT( Signals[Net.iNeuronOfst[TRIB]+index] ) 
							) ,
							( 
								CxAbsUT( Zs[Net.iNeuronOfst[LAY]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
							)
						)
					);
				}
			}
			/* layer is complete. */
			subkEvalSampleBetaMT( devSes, s, Net, true, Signals, Zs, Wt, XInputs, YEval, dYEval, devLearn.gpudSqrErr);
		}
	}
	/* correct output layer BP III.3 */
	long SUB = TOP-1; 
	//int iTopWidth=Net.iNeuronQTY[TOP];
	int iSubWidth=Net.iNeuronQTY[SUB];
			
	for (int k=1; k<Net.iNeuronQTY[TOP]; ++k){
		//for (int i=0; i<Net.iNeuronQTY[SUB]; ++i){  
		for (long offset=0; ( index =offset+tIx)< Net.iNeuronQTY[SUB] ; offset+=lTotalThreads){ // index stands for i
			/* For last layer only, adjustment to kth neuron's ith weight = k's delta * complex conjugate of i's signal / ( dendrites+1)  . */
						Wt[IDX2C( Net.iWeightOfst[TOP]+index, k, iSubWidth )]
			=CxAddCxUT( Wt[IDX2C( Net.iWeightOfst[TOP]+index, k, iSubWidth )] , 
				CxMultiplyCxUT( 
					Deltas[Net.iNeuronOfst[TOP]+k] , 
					CxConjugateUT( Signals[Net.iNeuronOfst[SUB]+index] ) 
				)
			);  // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
					//if(gDevDebug)if(k==1)		
					//	printf("<s%d.L1.k%d.i%d> Wt=% f+% f\n", s, k, index
					//	, Wt[IDX2C( Net.iWeightOfst[TOP]+index, k, iSubWidth )].x, Wt[IDX2C( Net.iWeightOfst[TOP]+index, k, iSubWidth )].y );
		}
	}
	/* backprop is complete. */
	/* re-evaluate sample to get new yields. BPIV */
//	subkEvalSampleBetaMT( s, Net, o, Signals, Zs, Wt, XInputs, YEval, dYEval);
//	 now, re-calculate delta-star for each output and ensure that value is zero (at least when alpha is 1) BPV
//	for(int i=0; i<Net.iNeuronQTY[TOP]; ++i){
//	for (long offset=0; ( index =offset+tIx)< Net.iNeuronQTY[TOP] ; offset+=lTotalThreads){ // index stands for i
//		  delta-star = D - Y = Desired output minus actual output from evaluation
//		  D is the cplx coords of the sector of the desired answer		Y is the complex result of evaluation of the given sample. */
//		Deltastar = CxSubtractCxUT( devLearn.gpuDOutputs[ IDX2C( index, s, ROWLEN ) ], 
//			Net.gpuSectorBdry[(int) dYEval[ IDX2C( index, s, ROWLEN ) ] ] );
//
//		 delta-star = D - Y = Desired output minus actual output from evaluation
//		double D =  devLearn.gpudDOutputs[ IDX2C( index, s, ROWLEN ) ];
//		double Y = dYEval[ IDX2C( index, s, ROWLEN ) ];
//			if(o)printf("sample %d output %d: %f > %f - %f\n", s, index, D-Y, D, Y );
//					if(o)if(s==2)
//						printf("i>%d D-Y > %f+%f - %f+%f\n", index, devLearn.gpuDOutputs[ IDX2C( index, s, ROWLEN ) ].x, devLearn.gpuDOutputs[ IDX2C( index, s, ROWLEN ) ].y 
//						, YEval[ IDX2C( index, s, ROWLEN ) ].x, YEval[ IDX2C( index, s, ROWLEN ) ].y );
//	}
//	return iReturn; /* number of weights updated. */
}


__device__ void subkBackPropEoptMT(long lSampleQtyReq, long o)
{/*! flags and counts samples meeting  */
	long row, OUTROWLEN=devLearn.iOutputQty+1; // prepare array index and width
	//long s = threadIdx.x + devSes.iEvalThreads * blockIdx.x; // s is thread index over the kernel
	long s = threadIdx.x + blockDim.x * blockIdx.x; // tIx is thread index over the kernel
	long lTotalThreads = gridDim.x * blockDim.x; // total number of threads
	//long lTotalThreads = devSes.iBpropThreads * devSes.iBpropBlocks; // total number of threads
	double maxSquared = devSes.dMAX * devSes.dMAX ; //needed to compare to stored delta squared values
	
	devlReturn[s]=0; // clear global mem accumulator; out of bound samples will remain at this value
	for (long k=0; (row=k+s)<lSampleQtyReq; k+=lTotalThreads){ // advance so-many samples each time, falling out if row is beyond bounds
		if( devLearn.gpudSqrErr[IDX2C( o, row, OUTROWLEN )] > maxSquared ){ // if the MAX criterion is exceeded	
			++devlReturn[s]; // increment the counter
		}
	}
	//end linear accumulation, begin intra-block reduction
							int j=blockDim.x/2;
							while (j){
								__syncthreads();
								if (threadIdx.x < j){ // select for bottom half of threads
									devlReturn[s] += devlReturn[s+j]; // add the top half values to their bottom half counterparts
								}
								j /=2; // divide bottom half into halves and do it again
							}
							__syncthreads();
							// all threads' values in a given block are now accumulated in its 0th thread's devdReturn[s]
							if(threadIdx.x==0){ // calling on each 0th thread
								atomicAdd( &devlTrainable, devlReturn[s] ); // accumulate each block's total atomically
							}
} // end of enumeration of backpropable samples


extern"C"
double knlFFeRmseOpt(struct rohanContext& rSes, long lSampleQtyReq, long o, char Option, int iBlocks, int iThreads)
{mIDfunc /*! checks sampled outputs vs evaluated outputs and returns root mean squared error. */
	// o is the index of output to be checked, or all outputs if o=0
	//Option will determine if existing data is used (E, not implemented) or refreshed (R) XX
	//
	//	externalities: devdRMSE in device-global scope
	//	calls: kernel mtkFFeRmseOptMT
	double dTotal=0.0; //float elapsedTime;
	
	cudaEvent_t start, stop;
	cudaEventCreate( &start);
	cudaEventCreate( &stop);
	
	// check if sample qty is outside the meaningful interval [1, all]
	if( (lSampleQtyReq<=0) || (lSampleQtyReq>rSes.rLearn->lSampleQty) )
		lSampleQtyReq=rSes.rLearn->lSampleQty; // default to all if so
	
		cudaEventRecord( start, 0);
	cudaMemcpyToSymbol( "devdRMSE", &dTotal, sizeof(double) );
		mCheckCudaWorked

	cudaPrintfInit();
	mtkFFeRmseOptMT<<< iBlocks , iThreads >>>( lSampleQtyReq, o, Option);
	cudaPrintfDisplay(rSes.deviceBucket, true);
	cudaPrintfEnd();

	cudaMemcpyFromSymbol( &dTotal, "devdRMSE", sizeof(double) );
		mCheckCudaWorked
		cudaEventRecord( stop, 0);
	cudaEventSynchronize( stop);
		cudaEventElapsedTime( &gElapsedTime, start, stop);
		gKernelTimeTally+=gElapsedTime;

	if(gDevDebug)conPrintf(">>DEVICE: Time to complete FFeRMSE kernel(%c): %3.1f ms\n", Option, gElapsedTime);
		cudaEventDestroy( start);
		cudaEventDestroy( stop);
		fprintf(rSes.deviceBucket, "RETURN: %f\t%f\n", dTotal, sqrt(dTotal/lSampleQtyReq) );
	return rSes.dDevRMSE=sqrt(dTotal/lSampleQtyReq);
}

__global__ void mtkFFeRmseOptMT( long lSampleQtyReq, long o, char Option)
{/*! checks sampled outputs vs evaluated outputs and calculates root mean squared error. */
	// lSampleQtyReq is the qty of samples to be used
	// o is the index of the desired output or zero for all
	// Option will determine if existing data is used (E, not implemented) or refreshed (R) XX
	//	or if will be single (R) or multiple (M) threads per sample
	//	(S) is single thread, no RMSE calc
	//
	//	externalities: devSes, devNet, devLearn, devdRMSE
	//	calls: subkEvalSampleSingleThread, subkEvalSampleBetaMT, subkRmseMT
	
	__shared__  __align__(16) cuDoubleComplex myWt[MAXWEIGHTS]; //each block will fill its own copy in faster shared memory
	long index, mindex, sindex, tIx = threadIdx.x + blockDim.x * blockIdx.x; // tIx is thread index over the kernel
	long lTotalThreads = gridDim.x * blockDim.x; // total number of threads
	
	for (long offset=0;	(index =offset+threadIdx.x)< MAXWEIGHTS; offset+=blockDim.x){ 
		myWt[index]=devNet.Wt[index]; //preloading weights reduces mass R RMSE/eval time from 3.0 ms to 2.6 ms: JAW 5/27/12
	}
	if(Option=='R' || Option == 'S'){
		//for (long s=0; s<lSampleQtyReq; s+=lIncrement){ // do this progressively, one thread per sample
		for (long offset=0; (sindex =offset+tIx)< lSampleQtyReq ; offset+=lTotalThreads){ // sindex stands for s
			cuDoubleComplex Signals[MAXNEURONS], Zs[MAXNEURONS];
			subkEvalSampleSingleThread(sindex, 'R', Signals, Zs, myWt, devLearn.gpuXInputs, devLearn.gpuYEval, devLearn.gpudYEval, devLearn.gpudSqrErr); // evaluate that sample with disposable Sums and retained Wt s
		}
	}
	else if(Option=='M'){
		for (long s=0; s<lSampleQtyReq; ++s){
			__shared__  __align__(16) cuDoubleComplex mySignals[MAXNEURONS], myZs[MAXNEURONS];
			for (long offset=0;	(mindex =offset+threadIdx.x)< MAXNEURONS; offset+=blockDim.x){ 
				mySignals[mindex]=devNet.Signals[mindex];
				myZs[mindex]=devNet.Zs[mindex]; // preloading weights reduces mass R RMSE/eval time from 3.0 ms to 2.6 ms, so we're going try it on the M's as well: JAW 5/27/12
			}
			subkEvalSampleBetaMT( devSes, s, devNet, true, mySignals, myZs, myWt, devLearn.gpuXInputs, devLearn.gpuYEval, devLearn.gpudYEval, devLearn.gpudSqrErr);
		}
	}
	// feed-forward evaluation above, RMSE calculation below
	if(Option != 'S'){
		subkRmseMTBeta(	lSampleQtyReq, o, devLearn.iOutputQty+1, devLearn.gpudSqrErr ); // after all samples are evaluated, do error calculations
		//if(tIx==0)devSes.dDevRMSE=sqrt(devdRMSE/lSampleQtyReq);
	}
}


__device__ void subkEvalSampleBetaMT(rohanContext& Ses, long s, rohanNetwork& Net, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval, double * dSqrErr)
{// Beta uses fixed length fields instead of nested pointer layers
	// delta squared is not updated, since they'll be updated when RMSE is checked at the end of a pass through the learning set
	//
	//	externalities: read devLearn
	//
	// this functions needs to work with single full warps only at this time 5/6/2012
	// M is for multiple threads per sample, typically as part of backprop

	long index, kindex; // for warpwise loops
	long tIx = threadIdx.x + blockDim.x * blockIdx.x; // tIx is thread index over the kernel
	long lTotalThreads = gridDim.x * blockDim.x; // total number of threads
	
	/*! layer zero (inputs) is special. */
	long INROWLEN=Net.iNeuronQTY[0];

	for (long offset=0; (index =offset+tIx)< MAXNEURONS ; offset+=lTotalThreads){ // index stands for i
		Signals[Net.iNeuronOfst[0]+index]= gpuZero;
		if(index < INROWLEN)
			Signals[Net.iNeuronOfst[0]+index]= XInputs[IDX2C( index, s, INROWLEN )];
	}
	__syncthreads();
	//for (long offset=0; (index =offset+tIx)< INROWLEN ; offset+=lTotalThreads){ // index stands for i
	//	Signals[Net.iNeuronOfst[0]+index]= XInputs[IDX2C( index, s, INROWLEN )];
	//}
	//__syncthreads();
	 /*! middle and top layers. */
	for (int L=1; L<Net.iLayerQty; ++L){
		//struct rohanLayer& lay = Net.rLayer[L];
		long LAY=L;
		int TRIB=L-1; // index of previous layer
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		//for (int k=0; k<iNeuronQTY; ++k){ //Neuron zero is not skipped, its output should be 1+0i as a check
		for (long offset=0; (kindex =offset+tIx)< iNeuronQTY ; offset+=lTotalThreads){ // kindex stands for k
			Zs[Net.iNeuronOfst[LAY]+kindex]=gpuZero;
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
			   			   Zs[Net.iNeuronOfst[LAY]+kindex] = 
				CxAddCxUT( Zs[Net.iNeuronOfst[LAY]+kindex] , 
					CxMultiplyCxUT(
						Wt[IDX2C( Net.iWeightOfst[LAY] + i, kindex, iSignalQTY )],
						Signals[Net.iNeuronOfst[TRIB]+i] ) ) ;
			}
			// ACTIVATE //
			Signals[Net.iNeuronOfst[LAY]+kindex] = CxActivateUT( Zs[Net.iNeuronOfst[LAY]+kindex], Net);
		}
	}
	__syncthreads();
	/*! last layer values are converted and stored here */
	long TOP = Net.iLayerQty-1;
	long OUTROWLEN=Net.iNeuronQTY[TOP];
	//for (int i=0; i<Net.iNeuronQTY[TOP]; ++i){ // continuous conversion begins here 
	for (long offset=0; (index =offset+tIx)< OUTROWLEN ; offset+=lTotalThreads){ // index stands for i
		YEval[IDX2C( index, s, OUTROWLEN )]
			= Signals[Net.iNeuronOfst[TOP]+index] ; // store final complex output(s)
		dYEval[IDX2C( index, s, OUTROWLEN )]
			= FUnitCxUT( YEval[IDX2C( index, s, OUTROWLEN )] ) * Net.iSectorQty; // convert final complex outputs to sectors and store that
		if(devLearn.iContOutputs==false) // round off decimal if disc outputs is set
			dYEval[IDX2C( index, s, OUTROWLEN )]
				= int(dYEval[IDX2C( index, s, OUTROWLEN )]);
		//
		///  Deltas not updated during backprop, which is what this eval method is for
		//
		//double dDelta=abs( devLearn.gpudDOutputs[IDX2C( index, s, OUTROWLEN )]-dYEval[IDX2C( index, s, OUTROWLEN )] ); 
		//if(dDelta>(double)(devNet.iSectorQty/2)) // if error arc is greater than half
		//	dDelta=((double)devNet.iSectorQty-dDelta); // set delta to the lesser arc length
		//dSqrErr[IDX2C( index, s, OUTROWLEN )]=dDelta*dDelta; // save delta squared
	}
	/*! end of sample evaluation. */
	__syncthreads();
}

__device__ void subkEvalSampleSingleThread(long s, char Option, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval, double * dSqrErr)
{	// This version of EvalSample is intended to run in parallel as widely as possible, with one thread per sample.
	// However, it can be run one-off for other purposes such as to verify other eval methods.
	// This is why the summation and weight arrays can be passed in via pointer, to allow them to be drawn from and returned to various sources.
	//
	//	externalities: read devNet, devLearn
	
	/*! here beginneth evaluation of sam. */
	/*! layer zero (inputs) is special. */
	/// virtual input neurons' outputs are network inputs converted to complex coords
	long INROWLEN=devLearn.iInputQty+1;
	for (int i=0; i<INROWLEN; ++i){
		Signals[devNet.iNeuronOfst[0]+i] = XInputs[IDX2C( i, s, INROWLEN )];
	}

	/*! middle and top layers. */
	for (int L=1; L<devNet.iLayerQty; ++L){
		long LAY=L;
		int TRIB=L-1; // index of previous layer
		int iNeuronQTY=devNet.iNeuronQTY[LAY]; // size of current layer
		int iSignalQTY=devNet.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=0; k<iNeuronQTY; ++k){ //Neuron zero is not skipped, its output should be 1+0i as a check
			Zs[devNet.iNeuronOfst[LAY]+k]=gpuZero; // init to zero before use
			for (int i=0; i<iSignalQTY; ++i) {//walk weights on inputs from previous layer
							Zs[devNet.iNeuronOfst[LAY]+k]
				=CxAddCxUT( Zs[devNet.iNeuronOfst[LAY]+k],
						CxMultiplyCxUT( 
							Wt[IDX2C( devNet.iWeightOfst[LAY] + i, k, iSignalQTY )], 
							Signals[devNet.iNeuronOfst[TRIB]+i]));
			}
			// ACTIVATE //
			Signals[devNet.iNeuronOfst[LAY]+k] = CxActivateUT( Zs[devNet.iNeuronOfst[LAY]+k] , devNet );
		}
	}
	 /*! last layer is also special  IMPORTANT keep synchronized with cuEvalSingleOutput in rohan-learn.cpp. */
	int iLastLayer=devNet.iLayerQty-1;
	long OUTROWLEN=devLearn.iOutputQty+1;
	//struct rohanLayer& top = devNet.gpuLayer[iLastLayer]; 
	long TOP = iLastLayer;

	for (int i=0; i<=devLearn.iOutputQty; ++i){ // continuous conversion begins here 
		YEval[IDX2C( i, s, OUTROWLEN )]
			= CxActivateUT( Signals[devNet.iNeuronOfst[TOP]+i] , devNet ); // apply activation function to final output(s) and store
		dYEval[IDX2C( i, s, OUTROWLEN )]
			=FUnitCxUT( YEval[IDX2C( i, s, OUTROWLEN )] )*devNet.iSectorQty; // convert activated final output to sectors and store that
			if(devLearn.iContOutputs==false) // round off decimal if disc activation is set
				//devLearn.gpudYEval[IDX2C( i, s, OUTROWLEN )]=int(devLearn.gpudYEval[IDX2C( i, s, OUTROWLEN )]);
				dYEval[IDX2C( i, s, OUTROWLEN )]=int(dYEval[IDX2C( i, s, OUTROWLEN )]);
			double dDelta=abs( devLearn.gpudDOutputs[IDX2C( i, s, OUTROWLEN )]-dYEval[IDX2C( i, s, OUTROWLEN )] ); 
			if(dDelta>(double)(devNet.iSectorQty/2)) // if error arc is greater than half
				dDelta=((double)devNet.iSectorQty-dDelta); // set delta to the lesser arc length
			dSqrErr[IDX2C( i, s, OUTROWLEN )]=dDelta*dDelta; // save delta squared
	}

	//if (devLearn.iContOutputs){
	//	for (int i=0; i<=devLearn.iOutputQty; ++i){ // continuous conversion begins here 
	//		devLearn.gpudYEval[IDX2C( i, s, OUTROWLEN )]= FUnitCxUT( Signals[devNet.iNeuronOfst[TOP]+i] )*devNet.iSectorQty; // cx output is converted to angle [0,1), then multiplied by k, then stored with sample
	//		devLearn.gpuYEval[IDX2C( i, s, OUTROWLEN )]= Signals[devNet.iNeuronOfst[TOP]+i] ; // unconverted cx output is also stored with sample
	//		double dDelta=abs( devLearn.gpudDOutputs[IDX2C( i, s, OUTROWLEN )]-devLearn.gpudYEval[IDX2C( i, s, OUTROWLEN )] ); 
	//		if(dDelta>(double)(devNet.iSectorQty/2)) // if error arc is greater than half
	//			dDelta=((double)devNet.iSectorQty-dDelta); // set delta to the lesser arc length
	//		devLearn.gpudSqrErr[IDX2C( i, s, OUTROWLEN )]=dDelta*dDelta; // save delta squared
	//	}
	//}
	//else{
	//	for (int i=0; i<=devLearn.iOutputQty; ++i){ // discrete conversion starts here
	//		devLearn.gpudYEval[IDX2C( i, s, OUTROWLEN )]=(double)floor(FUnitCxUT( Signals[devNet.iNeuronOfst[TOP]+i] )*devNet.iSectorQty); // cx output is converted to angle and multiplied by k, but then the fraciton is dropped before storing
	//		devLearn.gpuYEval[IDX2C( i, s, OUTROWLEN )]= Signals[devNet.iNeuronOfst[TOP]+i] ; // unconverted cx output is also stored with sample
	//		double dDelta=abs( devLearn.gpudDOutputs[IDX2C( i, s, OUTROWLEN )]-devLearn.gpudYEval[IDX2C( i, s, OUTROWLEN )] );
	//		if(dDelta>(double)(devNet.iSectorQty/2)) // if error arc is greater than half
	//			dDelta=((double)devNet.iSectorQty-dDelta); // set delta to the lesser arc length
	//		devLearn.gpudSqrErr[IDX2C( i, s, OUTROWLEN )]=dDelta*dDelta; // save delta squared
	//		// diagnostic print
	//		//if(s==2)printf("i=%d, Y=%f+%f, d= %f\n", i, devLearn.gpuYEval[IDX2C( i, s, OUTROWLEN )].x, devLearn.gpuYEval[IDX2C( i, s, OUTROWLEN )].y, dDelta );
	//	}
	//}
}


__device__ void subkRmseMT(long lSampleQtyReq, long o, int OUTROWLEN, double * dSqrErr)
{/*! sums all SE values for oth input */
	//	externalities 
	//				write devdReturn, devdRMSE
	//
	//	may need to run in full warp quantities of threads
	//	verified for 2 samples 5/23/12
	
	long row;//, OUTROWLEN=devLearn.iOutputQty+1; // prepare array index and width
	long tIx = threadIdx.x + blockDim.x * blockIdx.x; // tIx is thread index over the kernel
	long lTotalThreads = gridDim.x * blockDim.x; // total number of threads
	
	devdReturn[tIx]=0.0; // clear global mem accumulator; out of bound samples will remain at this value
	for (long k=0; (row=k+tIx)<lSampleQtyReq; k+=lTotalThreads){ // advance thread qty samples each time, falling out if row is beyond bounds
		devdReturn[tIx]+= dSqrErr[IDX2C( o, row, OUTROWLEN )]; // accumulate the delta squared for the indicated sample
		//if(gDevDebug) printf("[%d]=%f\t", tIx, dSqrErr[IDX2C( o, row, OUTROWLEN )]);
	}
	//end linear accumulation, begin intra-block reduction
	__syncthreads(); // crucial placement
	
	int j=blockDim.x/2;
	while (j){
		if (threadIdx.x < j || tIx+j < devLearn.lSampleQty){ // select for bottom half of threads in each block AND make sure that upper half is not beyond working samples
			devdReturn[tIx] += devdReturn[tIx+j]; // add the top half values to their bottom half counterparts
		}
		__syncthreads(); // crucial placement
		j /=2; // divide bottom half into halves and do it again
	}
	
	// all threads' values in a given block are now accumulated in its 0th thread's devdReturn[s]
	if(threadIdx.x==0){ // calling on each 0th thread
		atomicAdd(&devdRMSE, devdReturn[tIx]); // accumulate each block's total atomically
		//cuPrintf("devdReturn= %f\n", devdReturn[tIx]);
	}	
}

__device__ void subkRmseMTBeta(long lSampleQtyReq, long o, int OUTROWLEN, double * dSqrErr)
{/*! sums all SE values for oth input */
	//	externalities 
	//				write devdReturn, devdRMSE
	//
	//	may need to run in full warp quantities of threads
	//	verified for 2 samples 5/23/12
	
	__shared__ __align__(16)double dReturn[1024];
	long row;//, OUTROWLEN=devLearn.iOutputQty+1; // prepare array index and width
	long tIx = threadIdx.x + blockDim.x * blockIdx.x; // tIx is thread index over the kernel
	long lTotalThreads = gridDim.x * blockDim.x; // total number of threads
	
	dReturn[threadIdx.x]=0.0;
	//cuPrintf("threadIdx.x\t%d\tdReturn[]\t%f\n", threadIdx.x, dReturn[threadIdx.x]);
	//devdReturn[tIx]=0.0; // clear global mem accumulator; out of bound samples will remain at this value

	for (long k=0; (row=k+tIx)<lSampleQtyReq; k+=lTotalThreads){ // advance thread qty samples each time, falling out if row is beyond bounds
		//cuPrintf("k %d\tthreadIdx.%d\tdReturn[]\t%f\t+dSqrErr(%d,%d,%d)\t%f\t=\t%f\n", k, threadIdx.x, dReturn[threadIdx.x], o, row, OUTROWLEN, dSqrErr[IDX2C( o, row, OUTROWLEN )], dReturn[threadIdx.x] + dSqrErr[IDX2C( o, row, OUTROWLEN )]);
		cuPrintf("%d + %d = %d\n", k, tIx, k+tIx);
		dReturn[threadIdx.x]+= dSqrErr[IDX2C( o, row, OUTROWLEN )]; // accumulate the delta squared for the indicated sample
		//if(gDevDebug) printf("[%d]=%f\t", tIx, dSqrErr[IDX2C( o, row, OUTROWLEN )]);
	}
	//end linear accumulation, begin intra-block reduction
	__syncthreads(); // crucial placement
	
	int j=blockDim.x/2; // divide block of threads in half
	while (j){ // as long as half is still at least one
		if (threadIdx.x < j || tIx+j < lSampleQtyReq){ // select for bottom half AND make sure that upper half is not beyond working samples
			dReturn[threadIdx.x] += dReturn[threadIdx.x+j]; // add the top half values to their bottom half counterparts
			//devdReturn[tIx] += devdReturn[tIx+j]; // add the top half values to their bottom half counterparts
		}
		__syncthreads(); // crucial placement
		j /=2; // divide bottom half into halves and do it again
	}
	
	// all threads' values in a given block are now accumulated in its 0th thread's devdReturn[s]
	if(threadIdx.x==0){ // calling on each 0th thread
		atomicAdd(&devdRMSE, dReturn[0]); // accumulate each block's total atomically
		//cuPrintf("dReturn[0]= %f\n", dReturn[0]);
	}	
}


__device__ void subkShowMeDiffSums( cuDoubleComplex * Sums, char cSymbol, int x1, int x2, int x3)
{/// checks all elements of Sums against their corresponding XInputs and ZOutputs
	for(int L=1; L<devNet.iLayerQty; ++L){
		for (int i=0; i<=devNet.rLayer[L].iNeuronQty; ++i){
			double X = devNet.rLayer[L].ZOutputs[i].x - Sums[ devNet.iNeuronOfst[L]+i ].x;
			double Y = devNet.rLayer[L].ZOutputs[i].y - Sums[ devNet.iNeuronOfst[L]+i ].y;
			cuDoubleComplex delta = { X, Y };
			double Z = CxAbsUT(delta);
			if (Z>0.01){
				printf("host%c ZOutput %d,%d Sums %d position %d,%d,%d = %f,%f\n", cSymbol, L, i, devNet.iNeuronOfst[L]+1, x1, x2, x3, X+Y, Z);
				Sums[ devNet.iNeuronOfst[L]+i ] = devNet.rLayer[L].ZOutputs[i] ;
			}
		}
	}
	//return 0;
}


__device__ void subkShowMeResetSums( cuDoubleComplex * Sums)
{/// checks all elements of Sums against their corresponding XInputs and ZOutputs
	for(int L=0; L<devNet.iLayerQty; ++L){
		for (int i=0; i<=devNet.rLayer[L].iNeuronQty; ++i){
			Sums[ devNet.iNeuronOfst[L]+i ] = devNet.rLayer[L].ZOutputs[i] ;
		}
	}
	//return 0;
}

__device__ void subkEvalSingleSampleUT(long lSample)
{	/*! here beginneth evaluation of sam. */

	/*! layer zero (inputs) is special. */
		subkConvertInputsUT( lSample);
	 /*! middle and top layers. */
		subkEvalMidTopLayersUT( lSample);
	 /*! last layer is also special  IMPORTANT keep synchronized with cuEvalSingleOutput in rohan-learn.cpp. */
		subkOutputConvertUT( lSample);
	 /*! end of sample evaluation. */
}

__device__ void subkConvertInputsUT( long lSample)
{/// converts sample inputs to complex NN input layer values //sam refs removed 11/6
	/// layer zero (inputs) is special
	/// virtual input neurons' outputs are network inputs converted to complex coords
	//long s=lSample; //replace for-loop domain with requested sample index

	long ROWLEN=devLearn.iInputQty+1;
	
	for (int i=0; i<ROWLEN; ++i){
		devNet.gpuLayer[0].gpuZOutputs[i]=devLearn.gpuXInputs[IDX2C( i, lSample, ROWLEN )];


		//if(i==1)printf("subkCI%d| %g+%g -> %g+%g\n", lSample, devNet.gpuLayer[0].gpuZOutputs[i].x, devNet.gpuLayer[0].gpuZOutputs[i].y, devLearn.gpuXInputs[IDX2C( i, lSample, ROWLEN )].x, devLearn.gpuXInputs[IDX2C( i, lSample, ROWLEN )].y);


	}
	// end convert inputs
}

__device__ void subkEvalMidTopLayersUT( long lSample)
{/// number crunches the middle and top layers of an MLMVN 
	//const cuDoubleComplex gpuZero = { 0, 0 };
	//const cdcInit = { -999.0, 999.0 };

	for (int L=1; L<devNet.iLayerQty; ++L){
		//printf("subkEvalMidTopLayersUT Layer %d\n%dX|", L, L);
		//struct rohanLayer& lay = devNet.gpuLayer[L];
		int iLastNeuron=devNet.gpuLayer[L].iNeuronQty; // size of current layer
		int PL=L-1; // index of previous layer
		int iLastSignal=devNet.gpuLayer[L].iDendriteQty; // weight qty depends on size of previous layer
			//cuDoubleComplex*& wt = devNet.gpuLayer[L].gpuWeights;
			cuDoubleComplex*& oldOut = devNet.gpuLayer[PL].gpuZOutputs;
			cuDoubleComplex*& newOut = devNet.gpuLayer[L].gpuZOutputs;


		/*for (int j=0; j<=iLastSignal; ++j)
			printf("%f+%f,%d ", oldOut[j].x, oldOut[j].y, j);		
		printf("\n%dZ|", L);*/
			//printf("skEMTL%d| %g+%g ?= %g+%g\n", L, devNet.Wt[IDX2C(devNet.iWeightOfst[L]+1, 1, lay.iNeuronQty+1)].x, devNet.Wt[IDX2C(devNet.iWeightOfst[L]+1, 1, lay.iNeuronQty+1)].y, wt[IDX2C(1, 1, lay.iNeuronQty+1)].x, wt[IDX2C(1, 1, lay.iNeuronQty+1)].y);


		for (int i=0; i<=iLastNeuron; ++i){ //Neuron zero is not skipped, its output should be 1+0i as a check
			newOut[i]=gpuZero; //newOut[i].x=1; newOut[i].y=0;
			for (int j=0; j<=iLastSignal; ++j){ //walk weights on inputs from previous layer
				//if(i==1)printf("%g+%gX%g+%g\t", wt[IDX2C(i, j, lay.iNeuronQty+1)].x, wt[IDX2C(i, j, lay.iNeuronQty+1)].y, oldOut[j].x, oldOut[j].y);
				//newOut[i]=CxAddCxUT(newOut[i],CxMultiplyCxUT( wt[IDX2C(i, j, lay.iNeuronQty+1)] , oldOut[j]));
				newOut[i]=CxAddCxUT(newOut[i],CxMultiplyCxUT( devNet.Wt[IDX2C(devNet.iWeightOfst[L]+i, j, devNet.iNeuronQTY[L])], oldOut[j]));
			}


			//if(i==1)printf("\nskEMTL%d| %g+%g -> %g+%g\n", L, newOut[i].x, newOut[i].y, CxActivateUT(newOut[i]).x, CxActivateUT(newOut[i]).y);
			//if(i==1)printf("\n");

			// ACTIVATE //
			newOut[i]=CxActivateUT( newOut[i] , devNet );
		

			//printf("%f+%f,%d ", newOut[i].x, newOut[i].y, i);		
		
		
		}
		
		
		//printf("\n");
	
	
	}
	
	////end midtop layers
}


__device__ void subkOutputConvertUT(long lSample)
{/// converts complex NN output layer values to evaluated sample outputs //sam refs removed 11/6
	//long s=lSample; //replace for-loop domain with requested sample index
	int iLastLayer=devNet.iLayerQty-1;
	//struct rohanSample& sam = devLearn.rSample[s];
	long ROWLEN=devLearn.iOutputQty+1;
	struct rohanLayer& top = devNet.gpuLayer[iLastLayer];
	
	//printf("%ddev|", lSample);
	
	if (devLearn.iContOutputs){
		for (int i=0; i<=devLearn.iOutputQty; ++i){ // continuous conversion begins here 
			devLearn.gpudYEval[IDX2C( i, lSample, ROWLEN )]=FUnitCxUT(top.gpuZOutputs[i])*devNet.iSectorQty; // cx output is converted to angle [0,1), then multiplied by k, then stored with sample
			devLearn.gpuYEval[IDX2C( i, lSample, ROWLEN )]=top.gpuZOutputs[i]; // unconverted cx output is also stored with sample
			
			//printf("%g+%g\t",top.gpuZOutputs[i].x, top.gpuZOutputs[i].y);
		}
	}
	else{
		for (int i=0; i<=devLearn.iOutputQty; ++i){ // discrete conversion starts here
			devLearn.gpudYEval[IDX2C( i, lSample, ROWLEN )]=(double)floor(FUnitCxUT(top.gpuZOutputs[i])*devNet.iSectorQty); // cx output is converted to angle and multiplied by k, but then the fraciton is dropped before storing
			devLearn.gpuYEval[IDX2C( i, lSample, ROWLEN )]=top.gpuZOutputs[i];
		}
	}

	//printf("\n");

	// end output convert
}

__device__ double FUnitCxUT(const cuDoubleComplex A)
{/// returns the unitary angle of A in non-negative values [0,1)
	double fUnit;
	
	fUnit=atan2(A.y,A.x);
	fUnit*=ONE_OVER_TWO_PI;
	if(fUnit<0)
		++fUnit;

	return fUnit;
}

__device__ cuDoubleComplex CxAddCxUT(const cuDoubleComplex A, const cuDoubleComplex B)
{/// returns the sum of complex addends A and B
	cuDoubleComplex C;

	C.x = A.x + B.x;
	C.y = A.y + B.y;
	
	return C;
}

__device__ cuDoubleComplex CxMultiplyCxUT(const cuDoubleComplex A, const cuDoubleComplex B)
{/// returns product of complex factors A and B
	cuDoubleComplex C;

	// FOIL procedure for binomial multiplication: first, outside, inside, last
	// first is real times real, last is im times im
	C.x = A.x * B.x - A.y * B.y;
	// outside and inside are both real times imaginary
	C.y = A.x * B.y + A.y * B.x;

	return C;
}


__device__ cuDoubleComplex CxActivateUT(const cuDoubleComplex Z, rohanNetwork& Net)
{/// applies ContActivation or discrete activation function to cx neuron output and returns Phi(Z)
	/// GPU device vector based fn implemented 5/27/12
	//cuDoubleComplex phi;
	//if (Net.bContActivation) { // apply ContActivation activation function to weighted sum : phi(z)=z/|z|
	//	phi = CxDivideRlUT( Z, CxAbsUT( Z ) );
	//}
	//else {	// apply Discrete activation function to weighted sum : s=int(arctan(z)*k/2pi), phi(z)=(X(s),Y(s))
	//	double theta = atan2(Z.y, Z.x); // theta = arctan y/x
	//	int iSector = (int)((theta * Net.dK_DIV_TWO_PI) + Net.iSectorQty) % Net.iSectorQty;
	//	phi = Net.gpuSectorBdry[iSector];
	//}
	//return phi;
	return CxDivideRlUT( Z, CxAbsUT( Z ) );
}


__device__ cuDoubleComplex CxMultiplyRlUT(const cuDoubleComplex A, const double Rl)
{/// returns product of complex factor A and real factor Rl
	cuDoubleComplex B;

	B.x = Rl * A.x;
	B.y = Rl * A.y;
	
	return B;
}


__device__ cuDoubleComplex CxDivideRlUT(const cuDoubleComplex A, const double Rl)
{/// returns quotient of complex dividend A and real divisor Rl
	cuDoubleComplex B;
	double recip_Rl;

	recip_Rl = 1/Rl;
	B.x = A.x * recip_Rl; 
	B.y = A.y * recip_Rl;
	
	return B;
}

__device__ double CxAbsUT(const cuDoubleComplex Z)
{/// returns absolute value of Z; aka modulus or magnitude
	double abs;
	
	abs = sqrt( Z.x * Z.x + Z.y * Z.y ); 

	return abs;
}

__device__ cuDoubleComplex CxSubtractCxUT(const cuDoubleComplex A, const cuDoubleComplex B)
{/// returns difference of complex subends A and B
	cuDoubleComplex C;

	C.x = A.x - B.x;
	C.y = A.y - B.y;
	
	return C;
}

__device__ cuDoubleComplex CxDivideCxUT(const cuDoubleComplex A, const cuDoubleComplex B)
{/// returns quotient of complex dividend A and complex divisor B
	cuDoubleComplex C; double recip_denom;

	// (Ax + Ayi)/(Bx + Byi) is simplified by multiplying by the conjgate of B to 
	// (Ax + Ayi)*(Bx - Byi)/|B|^2
	recip_denom = 1 / (B.x * B.x + B.y * B.y); // this is 1/|B|^2
	// FOIL procedure for binomial multiplication: first, outside, inside, last
	// first is real times real, last is im times im
	C.x = A.x * B.x - A.y * (-B.y);
	// outside and inside are both real times imaginary
	C.y = A.x * (-B.y) + A.y * B.x;
	// now we apply the denominator
	C.x*=recip_denom;
	C.y*=recip_denom;
	// as seen on http://www.sosmath.com/complex/number/basic/soscv.html

	return C;
}

__device__ cuDoubleComplex CxConjugateUT(const cuDoubleComplex Z)
{/// returns complex conjugate of Z
	cuDoubleComplex C;

	C.x = Z.x;
	C.y = - Z.y;
	
	return C;
}

__device__ long d_answer;

extern "C" 
long knlCRC32Buf(char * buffer, unsigned int length)
{mIDfunc/// CRC32 code coutesy snippets.org
	
	long answer;
	mtkCRC32Buf<<< 1,  1 >>>(buffer, length);
	cudaMemcpyFromSymbol(&answer, "d_answer", sizeof(answer), 0, cudaMemcpyDeviceToHost);
	
	return answer;
}

__global__ __device__ void mtkCRC32Buf(char * buffer, unsigned int length)
{
	// d_answer is dev-global variable
	d_answer = subkCrc32buf(buffer, length);
	//cuPrintf("%08lX %7ld %s\n", d_answer, length, "from GPU");
}

typedef DWORD UNS_32_BITS;

__device__ long subkCrc32buf(char *buf, size_t len)
{
UNS_32_BITS crc_32_tab[] = { /* CRC polynomial 0xedb88320 */
0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};
      register DWORD oldcrc32;

      oldcrc32 = 0xFFFFFFFF;

      for ( ; len; --len, ++buf)
      {
            oldcrc32 = UPDC32(*buf, oldcrc32);
      }

      return ~oldcrc32;     
}


// function from CUDA C programming guide
__device__ double atomicAdd(double* address, double val) 
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address; 
	unsigned long long int old = *address_as_ull, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed))); 
	} 
	while (assumed != old); 
	return __longlong_as_double(old);
}

__device__ void __checksum(char * sLabel)
{
	if(threadIdx.x==0)printf("%s\n", sLabel );
	if(threadIdx.x==0)printf("Dall= %08lX\n", subkCrc32buf((char*)(devNet.Deltas), MAXNEURONS * 16) );
	if(threadIdx.x==0)printf("Sall= %08lX\n", subkCrc32buf((char*)(devNet.Signals), MAXNEURONS * 16) );
	if(threadIdx.x==0)printf("Wall= %08lX\n", subkCrc32buf((char*)(devNet.Wt), MAXWEIGHTS * 16) );
	if(threadIdx.x==0)printf("gD1== %08lX %d:%d\n", subkCrc32buf((char*)&devNet.Deltas[devNet.iNeuronOfst[1]], devNet.iNeuronQTY[1] * 16) , devNet.iNeuronOfst[1] , devNet.iNeuronQTY[1] );
	if(threadIdx.x==0)printf("gD2== %08lX %d:%d\n", subkCrc32buf((char*)&devNet.Deltas[devNet.iNeuronOfst[2]], devNet.iNeuronQTY[2] * 16) , devNet.iNeuronOfst[2] , devNet.iNeuronQTY[2] );
	if(threadIdx.x==0)printf("gS1== %08lX %d:%d\n", subkCrc32buf((char*)&devNet.Signals[devNet.iNeuronOfst[1]], devNet.iNeuronQTY[1] * 16) , devNet.iNeuronOfst[1] , devNet.iNeuronQTY[1] );
	if(threadIdx.x==0)printf("gS2== %08lX %d:%d\n", subkCrc32buf((char*)&devNet.Signals[devNet.iNeuronOfst[2]], devNet.iNeuronQTY[2] * 16) , devNet.iNeuronOfst[2] , devNet.iNeuronQTY[2] );
	if(threadIdx.x==0)printf("gW1== %08lX %d:%d\n", subkCrc32buf((char*)&devNet.Wt[devNet.iWeightOfst[1]], devNet.iNeuronQTY[1] * 16) , devNet.iWeightOfst[1] , devNet.iWeightQTY[1] );
	if(threadIdx.x==0)printf("gW2== %08lX %d:%d\n", subkCrc32buf((char*)&devNet.Wt[devNet.iWeightOfst[2]], devNet.iWeightQTY[2] * 16) , devNet.iWeightOfst[2] , devNet.iWeightQTY[2] );
}
