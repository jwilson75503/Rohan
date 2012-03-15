/* Includes, cuda */
#include "stdafx.h"

#define _USE_MATH_DEFINES
#define ONE_PI 3.14159265358979323846264338327950288
#define TWO_PI 6.283185307179586476925286766558
#define TWO_PI_OVER_384 0.01636246173744683978365960095458


//#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))

extern int iDebugLvl, iTrace;


int cuEvalSingleSample16(struct rohanContext& rSes, long lSample)
{mIDfunc
	int iMethodsUsed=0;
	 /*! here beginneth evaluation of sam. */
	cuDoubleComplex Sums[2048];
	 /*! layer zero (inputs) is special. */
		cuConvertInputs16(rSes, lSample, Sums);
	 /*! middle and top layers. */
		cuEvalMidTopLayers16(rSes, lSample, Sums);
	 /*! last layer is also special  IMPORTANT keep synchronized with cuEvalSingleOutput in rohan-learn.cpp. */
		cuOutputConvert16(rSes, lSample, Sums);
	 /*! end of sample evaluation. */
		++iMethodsUsed;

	return iMethodsUsed;
}

//int cuEvalSingleSample(struct rohanContext& rSes, long lSample)
//{mIDfunc
//	int iMethodsUsed=0;
//	 /*! here beginneth evaluation of sam. */
//	 /*! layer zero (inputs) is special. */
//		cuConvertInputs(rSes, lSample);
//	 /*! middle and top layers. */
//		cuEvalMidTopLayers(rSes, lSample);
//	 /*! last layer is also special  IMPORTANT keep synchronized with cuEvalSingleOutput in rohan-learn.cpp. */
//		cuOutputConvert(rSes, lSample);
//	 /*! end of sample evaluation. */
//		++iMethodsUsed;
//
//	return iMethodsUsed;
//}


int cuEvalSingleSampleBeta(struct rohanContext& Ses, long s, rohanNetwork& Net, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{// Beta uses fixed length fields instead of nested pointer layers
	// o is currently used to control diagnostic output
	 /*! layer zero (inputs) is special. */
	long INROWLEN=Net.iNeuronQTY[0];//rSes.rLearn->iInputQty+1;
	for (int i=0; i<INROWLEN; ++i){
		Signals[Net.iNeuronOfst[0]+i]= XInputs[IDX2C( i, s, INROWLEN )];
		//if(s==2 && o)
		//	printf("s>%d i>%d X>%f+%f\n", s, i, Signals[Net.iNeuronOfst[0]+i].x, Signals[Net.iNeuronOfst[0]+i].y);
	}
	 /*! middle and top layers. */
	for (int L=1; L<Net.iLayerQty; ++L){
		//struct rohanLayer& lay = Net.rLayer[L];
		long LAY=L;
		int TRIB=L-1; // index of previous layer
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=0; k<iNeuronQTY; ++k){ //Neuron zero is not skipped, its output should be 1+0i as a check
			Zs[Net.iNeuronOfst[LAY]+k]=cdcZero;
			
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
			   			 Zs[Net.iNeuronOfst[LAY]+k] = 
				CxAddCx( Zs[Net.iNeuronOfst[LAY]+k] , 
					CxMultiplyCx(
						//wt[IDX2C( i, k, Net.iDendrtQTY[LAY])], 
						Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )],
						Signals[Net.iNeuronOfst[TRIB]+i] ) ) ;
				//printf("k=%d i=%d wt= %f+%f\n", k, i, 
				//	Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )].x,
				//	Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )].y );
				//if ( Zs[Net.iNeuronOfst[LAY]+k].x == cdcZero.x && o)
				//	printf("Z=0!! k=%d i=%d\n", k, i);
				//if(s==2 && o)
				//	printf("k=%d i=%d Z=% f+% f", k,  i, Zs[Net.iNeuronOfst[LAY]+k].x, Zs[Net.iNeuronOfst[LAY]+k].y);
				//if(s==2 && o)
				//	printf(" Wt=% f+% f, X=% f+% f\n", Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )].x, Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )].y,
				//	Signals[Net.iNeuronOfst[TRIB]+i].x, Signals[Net.iNeuronOfst[TRIB]+i].y);
				//if (wt[IDX2C( i, k, Net.iDendrtQTY[LAY])].x-Wt[IDX2C( i, k, iSignalQTY )].x +
				//	wt[IDX2C( i, k, Net.iDendrtQTY[LAY])].y-Wt[IDX2C( i, k, iSignalQTY )].y)
				//	printf("weight mismatch: s%d L%d k%d i%d\n", s, L, k, i );

			}
	
			// ACTIVATE //
			Signals[Net.iNeuronOfst[LAY]+k] = CxActivate( Zs[Net.iNeuronOfst[LAY]+k] , Ses );
			//if(s==2)printf("k>%d phi>%f+%f, Z> %f+%f\n", k, Net.Signals[Net.iNeuronOfst[LAY]+k].x, Net.Signals[Net.iNeuronOfst[LAY]+k].y, 
			//	Net.Zs[Net.iNeuronOfst[LAY]+k].x, Net.Zs[Net.iNeuronOfst[LAY]+k].y );
			//if((Net.Signals[Net.iNeuronOfst[LAY]+k].x + Net.Signals[Net.iNeuronOfst[LAY]+k].y)>1000000)
			//	printf(": bad value from Signals[%d] !\n", Net.iNeuronOfst[LAY]+k);
		}
	}
	 
	/*! last layer values are converted and stored here */
	long TOP = Net.iLayerQty-1;
	long OUTROWLEN=Net.iNeuronQTY[TOP];
	
	for (int i=0; i<OUTROWLEN; ++i){ // continuous conversion begins here 
		YEval[IDX2C( i, s, OUTROWLEN )]= Signals[Net.iNeuronOfst[TOP]+i] ; // store final complex output(s)
		dYEval[IDX2C( i, s, OUTROWLEN )]=FUnitCx( YEval[IDX2C( i, s, OUTROWLEN )] ) * Net.iSectorQty; // convert final complex outputs to sectors and store that
		if(Ses.rLearn->iContOutputs==false) // round off decimal if disc activation is set
			dYEval[IDX2C( i, s, OUTROWLEN )]=floor(dYEval[IDX2C( i, s, OUTROWLEN )]);
		//if(o)if(s==2)
		//			printf("OUT%d F=% f+% f , FU=% f, Y=% f\n", i, YEval[IDX2C( i, s, OUTROWLEN )].x, YEval[IDX2C( i, s, OUTROWLEN )].y, 
		//			FUnitCx( YEval[IDX2C( i, s, OUTROWLEN )] ), dYEval[IDX2C( i, s, OUTROWLEN )] );
	}
	 /*! end of sample evaluation. */
	return true;
}


int cuEvalSingleOutput(rohanContext& rSes, long lSampleIdxReq, int iOutputIdxReq)
{mIDfunc/*! This will apply an MLMVN weight set to a given sample of a learning set and record the resulting final output for each.
 /*! ! Discrete inputs and outputs are used. Real integers are converted via K-valued logic to complex coordinates,
 /*! ! which are then product-summed by successive layers of neurons, then conveted back to integer output
 /*! !
 /*! ! IMPORTANT: keep this code consistent with cuEvalNNLearnSet in rohan-data.cpp. */
	long ROWLEN = rSes.rLearn->iOutputQty+1;

	long s=lSampleIdxReq /*! replace for-loop domain with requested sample index. */;
	double two_pi_div_sect_qty = TWO_PI/rSes.rNet->iSectorQty;
	//struct rohanSample& sam = rSes.rLearn->rSample[s];
 	cuEvalSingleSampleBeta(rSes, s, *rSes.rNet, 0, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval);
	int i=iOutputIdxReq; // replace for-loop domain with requested output index
	return (int)floor(rSes.rLearn->dYEval[ IDX2C( i, s, ROWLEN ) ]); /*! return evaluated result. */
}	

long OutputValidate(rohanContext& rSes)
{mIDfunc  /*! compares outputs by sample and by method to verify bulk outputs.. */
	int iReturn=0;
	long ROWLEN = rSes.rLearn->iOutputQty+1;
	
	for(long s=0; s<rSes.rLearn->lSampleQty; ++s){
		for(int i=0; i<=rSes.rLearn->iOutputQty; ++i){
			//printf("%d %d %f %f\t", s, i, rSes.rLearn->dYEval[ IDX2C( s, i, ROWLEN ) ] , rSes.rLearn->dAltYEval[ IDX2C( s, i, ROWLEN ) ] );
			if(rSes.rLearn->iContOutputs){
				if( abs(rSes.rLearn->dYEval[ IDX2C( i, s, ROWLEN ) ] - rSes.rLearn->dAltYEval[ IDX2C( i, s, ROWLEN ) ] ) >0.5 )
					++iReturn;
			}
			else{
				if( rSes.rLearn->dYEval[ IDX2C( i, s, ROWLEN ) ] - rSes.rLearn->dAltYEval[ IDX2C( i, s, ROWLEN ) ] )
					++iReturn;
			}
		}
	}
	return iReturn; // return number of outputs that diverged
}


int devResetAllDeltasAndOutputs(rohanContext& rSes)
{mIDfunc
	//knlResetDeltasOutputs(rSes);
	return 0;
}

int cuResetAllDeltasAndOutputs(rohanContext& rSes)
{mIDfunc
	for (int L=1; L<rSes.rNet->iLayerQty; ++L)  /*! reset outputs and deltas for full neuron layers. */
		for (int i = 0; i<=rSes.rNet->rLayer[L].iNeuronQty; ++i){
			rSes.rNet->rLayer[L].Deltas[i]=cdcZero;
			rSes.rNet->rLayer[L].ZOutputs[i]=cdcZero;
		}
	for (int i = 0; i< MAXNEURONS; ++i){
		rSes.rNet->Deltas[i]=cdcZero;
		rSes.rNet->Signals[i]=cdcZero;
		rSes.rNet->Zs[i]=cdcZero;
	}
	
	return 0;
}


int cuBackpropSingleSample(rohanContext& rSes, long s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{ mIDfunc /*! propagates adjustment of weights backwards preceeding layers from the chosen network output. */
	// s is sample index
	int iReturn=0 /* returns number of weights adjusted */ ;
	long ROWLEN = rSes.rLearn->iOutputQty+1;
	int o=false; // true=print intermediate eval figures
	/* clear all temp values BP0 */
	cuResetAllDeltasAndOutputs(rSes);
	/* re-evaluate sample to load temp values. BPI */
	cuEvalSingleSampleBeta(rSes, s, Net, false, Signals, Zs, Wt, XInputs, YEval, dYEval);
//if(o)printf("begin error calculation...\n");
	/* begin error calculation. BPII */
	cuDoubleComplex Deltastar /* measured error at the chosen network output. */ ;
	long TOP=Net.iLayerQty-1;
	/* calc top layer deltas. */
	for(int i=0; i<Net.iNeuronQTY[TOP]; ++i){
		 /* delta-star = D - Y = Desired output minus actual output from evaluation
		 /* D is the cplx coords of the sector of the desired answer		Y is the complex result of evaluation of the given sample, unactivated. */
		Deltastar = CxSubtractCx( 
						rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ], 
						Signals[Net.iNeuronOfst[TOP]+i] );
					//if(s==2 && o )
					//	printf("i>%d D-Y > %f+%f - ", i, rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].x, rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].y );
					//if(s==2 && o )
					//	printf("%f+%f\n", Signals[Net.iNeuronOfst[TOP]+i].x, Signals[Net.iNeuronOfst[TOP]+i].y );
		 /* divide the correction; delta = alpha * delta-star / n+1 (but alpha is always 1 for now). */
		Deltas[Net.iNeuronOfst[TOP]+i] = CxDivideRl( Deltastar, Net.iDendrtQTY[TOP] );
					//if(s==2 && o )
					//		printf("i>%d d*> %f+%f /%d> ", i, Deltastar.x, Deltastar.y , Net.iDendrtQTY[TOP] );
					//if(s==2 && o )
					//	printf("%f+%f > d\n", Deltas[Net.iNeuronOfst[TOP]+i].x, Deltas[Net.iNeuronOfst[TOP]+i].y );
	}
//if(o)printf("Distribute correction...\n");
	/* Now distribute the correction to lower layers if any. BPII.1 */
	if (Net.iLayerQty>2){  /* remember layer 0 = inputs, layer 1 = bottom row, layer {2..iLayerQty-2} = middle row, layer iLayerQty-1 = top row. */
		for (int L=Net.iLayerQty-1; L>1; --L){
			long LAY = L; /* setup access to layers. */
			long TRIB = L-1; /* trib for tributary.*/
			int iTributQTY=Net.iNeuronQTY[TRIB];
			int Sj=Net.iDendrtQTY[TRIB]; if (TRIB==1) Sj=1; // Sj=1 for firest hidden layer
			for (int i=1; i<Net.iNeuronQTY[LAY]; ++i) { // skip 0th neuron as its weights are either 1 (div identity) or 0 (div forbidden) and don't change anyway
				// k index must begin at 1, neuron zero not valid for correction
				for (int k=1; k<iTributQTY; ++k) { /* the contribution to ith neuron's kth tributary's delta = i's delta/i's weight k. */
								Deltas[Net.iNeuronOfst[TRIB]+k] 
					= CxAddCx ( Deltas[Net.iNeuronOfst[TRIB]+k] , 
						CxDivideCx( 
							Deltas[Net.iNeuronOfst[LAY]+i] , 
							Wt[IDX2C( Net.iWeightOfst[LAY]+k, i, iTributQTY )] ));
				//if(s==2 && o)			
				//	printf("TRIB %d k=%d i=%d d%f+%f / Wt%f+%f > %f+%f\n", TRIB, k,  i, Deltas[Net.iNeuronOfst[LAY]+i].x, Deltas[Net.iNeuronOfst[LAY]+i].y
				//	, Wt[IDX2C( Net.iWeightOfst[LAY]+k, i, iTributQTY )].x, Wt[IDX2C( Net.iWeightOfst[LAY]+k, i, iTributQTY )].y,
				//	Deltas[Net.iNeuronOfst[TRIB]+k].x, Deltas[Net.iNeuronOfst[TRIB]+k].y);
				}
			}
			// k index must begin at 1, neuron zero not valid for correction
			for (int k=1; k<iTributQTY; ++k) { /* contributions accumulated, now divide by dendrites+1. */
//				cuDoubleComplex preDiv=Deltas[Net.iNeuronOfst[TRIB]+k]; // diagnostic purpose only, remove if removing other diags
					//if(s==2 && o )
					//	printf("...k>%d d> %f+%f /%d> ", k, Deltas[Net.iNeuronOfst[TRIB]+k].x, Deltas[Net.iNeuronOfst[TRIB]+k].y , Sj );
				Deltas[Net.iNeuronOfst[TRIB]+k] 
					= CxDivideRl( 
						Deltas[Net.iNeuronOfst[TRIB]+k] , 
						Sj );
//					if(s==2 && o)
//						printf("...k>%d d> %f+%f /%d> %f+%f\n", k, preDiv.x, preDiv.y, Sj, Deltas[Net.iNeuronOfst[TRIB]+k].x, Deltas[Net.iNeuronOfst[TRIB]+k].y);
			}
		}
	}
	/* error distribution completed */
//if(o)printf("Error distribution completed.\n");
	/* and now update the weights BP III */
	/* adj weights on first hidden layer. */
		int FHID = 1;
		int SIG = 0;
		int iSignalQTY=rSes.rLearn->iInputQty+1;
		int iHidWidth=Net.iNeuronQTY[FHID];
	for (int k=1; k<iHidWidth; ++k){
		for (int i=0; i<iSignalQTY; ++i){  
			/* dW=d*xbar/s1/|z|= neuron's delta * input's conjugate / ( dendrites+1 * abs of input i ). */
					  Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )]
			=CxAddCx( Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )] , 
				CxDivideRl( 
					CxMultiplyCx( 
						Deltas[Net.iNeuronOfst[FHID]+k] , 
						CxConjugate( Signals[Net.iNeuronOfst[SIG]+i] ) 
					) , 
					CxAbs( Zs[Net.iNeuronOfst[FHID]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
				)
			);
				//if(s==2 && o )			
				//	printf("k=%d i=%d z=% f+% f", k,  i, Zs[Net.iNeuronOfst[FHID]+k].x, Zs[Net.iNeuronOfst[FHID]+k].y);
				//if(s==2 && o)
				//	printf(" Wt=% f+% f, X=% f+% f\n", Wt[IDX2C( Net.iWeightOfst[FHID] + i, k, iSignalQTY )].x, Wt[IDX2C( Net.iWeightOfst[FHID] + i, k, iSignalQTY )].y,
				//	Signals[Net.iNeuronOfst[SIG]+i].x, Signals[Net.iNeuronOfst[SIG]+i].y);
		}
		//printf("> %08lX\n", crc32buf((char*)(Net.Wt), MAXWEIGHTS * 16) );
	}
	//printf(">> %08lX\n", crc32buf((char*)(Net.Wt), MAXWEIGHTS * 16) );
	/* re-evaluate sample to update temp values. */
//	if(s==2 && o )
//		printf("first hidden wts adjusted...\n");
	cuEvalSingleSampleBeta(rSes, s, Net, false, Signals, Zs, Wt, XInputs, YEval, dYEval);
	if (Net.iLayerQty>2){
		 /* now use those outputs' conjugates and the deltas to adjust middle layers. BP III.1 */
		for (int L=2; L<Net.iLayerQty-1; ++L){
			 /* setup access to layers. */
			//struct rohanLayer& lay = Net.rLayer[L]; 
			long LAY = L;
			//struct rohanLayer& trib = Net.rLayer[L-1] /* trib for tributary. */ ; 
			long TRIB = L-1;
			int iLayWidth=Net.iNeuronQTY[LAY];
			int iTribWidth=Net.iNeuronQTY[TRIB];
			for (int k=1; k<Net.iNeuronQTY[LAY]; ++k){
				for (int i=0; i<Net.iNeuronQTY[TRIB]; ++i){  
					/* the adjustment added to kth neuron's ith trib's weight = k's delta * complex conjugate of i's signal / (abs of k's previous-wt product-sum * dendrites+1)  . */
							  Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )]
					=CxAddCx( Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )] , 
						CxDivideRl( 
							CxMultiplyCx( 
								Deltas[Net.iNeuronOfst[LAY]+k] , 
								CxConjugate( Signals[Net.iNeuronOfst[TRIB]+i] ) 
							) ,
							( 
								CxAbs( Zs[Net.iNeuronOfst[LAY]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
							)
						)
					);
				}
			}
			/* layer is complete. */
//			if(o)printf("Layer %d adjusted...\n", LAY);
			cuEvalSingleSampleBeta(rSes, s, Net, true, Signals, Zs, Wt, XInputs, YEval, dYEval);
		}
	}
	/* correct output layer BP III.3 */
	long SUB = TOP-1; 
	int iTopWidth=Net.iNeuronQTY[TOP];
	int iSubWidth=Net.iNeuronQTY[SUB];
			
	for (int k=1; k<Net.iNeuronQTY[TOP]; ++k){
		for (int i=0; i<Net.iNeuronQTY[SUB]; ++i){  
			/* For last layer only, adjustment to kth neuron's ith weight = k's delta * complex conjugate of i's signal / ( dendrites+1)  . */
					  Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )]
			=CxAddCx( Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )] , 
				CxMultiplyCx( 
					Deltas[Net.iNeuronOfst[TOP]+k] , 
					CxConjugate( Signals[Net.iNeuronOfst[SUB]+i] ) 
				)
			);  // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
		}
	}
	/* backprop is complete. */

//if(o)printf("Final layer adjusted...\n");
//	/* re-evaluate sample to get new yields. BPIV */
//	cuEvalSingleSampleBeta(rSes, s, Net, o, Signals, Zs, Wt, XInputs, YEval, dYEval);
//	// now, re-calculate delta-star for each output and ensure that value is zero (at least when alpha is 1) BPV
//	
//
//	for(int i=0; i<Net.iNeuronQTY[TOP]; ++i){
//
//		 // delta-star = D - Y = Desired output minus actual output from evaluation
//		 // D is the cplx coords of the sector of the desired answer		Y is the complex result of evaluation of the given sample. */
//		Deltastar = CxSubtractCx( rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ], 
//			Net.cdcSectorBdry[(int) dYEval[ IDX2C( i, s, ROWLEN ) ] ] );
//
//		if(o)printf("after %d: % 1f+% 1fi > % 1f+% 1fi - % 1f+% 1fi\n(%f)\n", i, Deltastar.x, Deltastar.y, 
//			  rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].x, 
//			  rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].y,
//			  Net.cdcSectorBdry[(int) dYEval[ IDX2C( i, s, ROWLEN ) ] ].x, 
//			  Net.cdcSectorBdry[(int) dYEval[ IDX2C( i, s, ROWLEN ) ] ].y, 
//			  dYEval[ IDX2C( i, s, ROWLEN ) ] );
//		// delta-star = D - Y = Desired output minus actual output from evaluation
//		double D =  rSes.rLearn->dDOutputs[ IDX2C( i, s, ROWLEN ) ];
//		double Y = dYEval[ IDX2C( i, s, ROWLEN ) ];
//			if(o)printf("sample %d output %d: %f > %f - %f\n", s, i, D-Y, D, Y );	
//					if(o)if(s==2)
//						printf("i>%d D-Y > %f+%f - %f+%f\n", i, rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].x, rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].y 
//						, YEval[ IDX2C( i, s, ROWLEN ) ].x, YEval[ IDX2C( i, s, ROWLEN ) ].y );
//	}
	
	return iReturn; /* number of weights updated. */
}

int TrainNNThresh(struct rohanContext& rSes, long bChangeWeights)
{mIDfunc 
/*! checks sampled outputs vs evaluated outputs, and returns number of samples that exceed threshold
 *  excessive samples are submitted for backpropagation if bChangeWeights is true.
 */
	int iReturn=0;
	double dDelta=0;
	long ROWLEN = rSes.rLearn->iOutputQty+1 ;

	if (rSes.lSampleQtyReq<=0 || rSes.lSampleQtyReq>rSes.rLearn->lSampleQty)
		rSes.lSampleQtyReq=rSes.rLearn->lSampleQty;
	//adjust requested amount to available values
	for(long s=0; s<rSes.lSampleQtyReq; ++s){  // loop over samples.
		//struct rohanSample& sam = rSes.rLearn->rSample[s];
		int iOverMAX=0;
		for(int i=0; i<=rSes.rLearn->iOutputQty; ++i){  // loop over outputs.
			dDelta = (double) abs( rSes.rLearn->dDOutputs[ IDX2C( i, s, ROWLEN ) ] - rSes.rLearn->dYEval[ IDX2C( i, s, ROWLEN ) ] );
			 // printf("dDelta %f dDelta*2 %f, Sectors %d\n", dDelta, dDelta*2, Net.iSectorQty);
			if((dDelta*2)>rSes.rNet->iSectorQty)
				dDelta=rSes.rNet->iSectorQty-dDelta;
			 // printf("Sample %d, output %f eval %f delta %f\n", s, sam.dXInputs[rSes.rLearn->iInputQty+i], sam.dYEval[i], dDelta);
			if( dDelta > rSes.dMAX)  // if effective error exceeds MAX, make a note
				++iOverMAX;
		}
		if (iOverMAX!=0) {	 // if a note has been made. 
			++iReturn; // increment the number of excessive samples.
			if (bChangeWeights) {  // and correct weights if that is desired.
				cuBackpropSingleSample(rSes, s, *rSes.rNet, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rNet->Deltas, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval);
			}
		}
	}
	return (iReturn);
}


double RmseNN(struct rohanContext& rSes, long lSampleQtyReq)
{mIDfunc /*! checks sampled outputs vs evaluated outputs and calculates root mean squared error. */
	double dReturn=0;
	FILE *fShow=rSes.debugHandle;
	// check if sample qty is outside the meaningful interval [1, all]
	if(lSampleQtyReq<=0 || lSampleQtyReq>rSes.rLearn->lSampleQty)
		lSampleQtyReq=rSes.rLearn->lSampleQty; // default to all if so
	for(long s=0; s<lSampleQtyReq; ++s){
		//struct rohanSample& sam = rSes.rLearn->rSample[s]; // loop over all requested samples and documented outputs
		for(int i=1; i<=rSes.rLearn->iOutputQty; ++i){
			double dDelta = (double)abs( rSes.rLearn->dDOutputs[IDX2C( i, s, (rSes.rLearn->iOutputQty+1))] - rSes.rLearn->dYEval[IDX2C( i, s, (rSes.rLearn->iOutputQty+1))] ); // delta = Desired - Yielded values
			if(dDelta>(double)(rSes.rNet->iSectorQty/2)) 
				dDelta=((double)rSes.rNet->iSectorQty-dDelta); // set delta to the lesser arc length
			dReturn+=(dDelta*dDelta); // accumulate squared error 
		}
	}
	//printf(">RmseNN total %f\n", dReturn);
	dReturn=sqrt(dReturn/(double)(lSampleQtyReq*rSes.rLearn->iOutputQty)); // take the root of the mean of the accumulated square error
	//printf(">>>RmseNN %f\n", dReturn);
	return dReturn;
}


void cuCksum(struct rohanContext& rSes)
{
	printf("Dall> %08lX\n", crc32buf((char*)(rSes.rNet->Deltas), MAXNEURONS * 16) );
	printf("Sall> %08lX\n", crc32buf((char*)(rSes.rNet->Signals), MAXNEURONS * 16) );
	printf("Wall> %08lX\n", crc32buf((char*)(rSes.rNet->Wt), MAXWEIGHTS * 16) );
	printf("D1>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Deltas[0+rSes.rNet->iNeuronOfst[1]], rSes.rNet->iNeuronQTY[1] * 16) , rSes.rNet->iNeuronOfst[1] , rSes.rNet->iNeuronQTY[1] );
	printf("D2>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Deltas[0+rSes.rNet->iNeuronOfst[2]], rSes.rNet->iNeuronQTY[2] * 16) , rSes.rNet->iNeuronOfst[2] , rSes.rNet->iNeuronQTY[2] );
	printf("S1>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Signals[0+rSes.rNet->iNeuronOfst[1]], rSes.rNet->iNeuronQTY[1] * 16) , rSes.rNet->iNeuronOfst[1] , rSes.rNet->iNeuronQTY[1] );
	printf("S2>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Signals[0+rSes.rNet->iNeuronOfst[2]], rSes.rNet->iNeuronQTY[2] * 16) , rSes.rNet->iNeuronOfst[2] , rSes.rNet->iNeuronQTY[2] );
	printf("W1>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Wt[0+rSes.rNet->iWeightOfst[1]], rSes.rNet->iWeightQTY[1] * 16) , rSes.rNet->iWeightOfst[1] , rSes.rNet->iWeightQTY[1] );
	printf("W2>> %08lX %d:%d\n", crc32buf((char*)&rSes.rNet->Wt[0+rSes.rNet->iWeightOfst[2]], rSes.rNet->iWeightQTY[2] * 16) , rSes.rNet->iWeightOfst[2] , rSes.rNet->iWeightQTY[2] );
	
}