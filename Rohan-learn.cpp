/* Includes, cuda */
#include "stdafx.h"

extern int gDebugLvl, gDevDebug, gTrace;


int cuEvalSingleSampleBeta(struct rohanContext& Ses, long s, rohanNetwork& Net, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{// Beta uses fixed length fields instead of nested pointer layers
	// o is currently used to control diagnostic output
	 /*! layer zero (inputs) is special. */
	long INROWLEN=Net.iNeuronQTY[0];//rSes.rLearn->iInputQty+1;
	for (int i=0; i<INROWLEN; ++i){
		Signals[Net.iNeuronOfst[0]+i]= XInputs[IDX2C( i, s, INROWLEN )];
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
						Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )],
						Signals[Net.iNeuronOfst[TRIB]+i] ) ) ;
			}
			// ACTIVATE //
			Signals[Net.iNeuronOfst[LAY]+k] = CxActivate( Zs[Net.iNeuronOfst[LAY]+k] , Net );
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
	}
	 /*! end of sample evaluation. */
	return true;
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


int cuBackpropLearnSet(rohanContext& rSes, long lSampleQtyReq, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{ mIDfunc /*! propagates adjustment of weights backwards preceeding layers from the chosen network output. */
	// lSampleQty is sample qty requested
	long lSubmitted=0;
	if(lSampleQtyReq < 1 || lSampleQtyReq > rSes.rLearn->lSampleQty) // if requested qty is out of bounds, use max
		lSampleQtyReq=rSes.rLearn->lSampleQty;
	for(long s=0; s<rSes.rLearn->lSampleQty; ++s){ // submit all samples requestee, one at a time
		cuBackpropSingleSample(rSes, s,  Net, Signals, Zs, Wt, Deltas, XInputs, YEval, dYEval );
		++lSubmitted;
	}

	return lSubmitted; // return qty of samples submitted
}


int cuBackpropSingleSample(rohanContext& rSes, long s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{ mIDfunc /*! propagates adjustment of weights backwards preceeding layers from the chosen network output. */
	// s is sample index
	int iReturn=0 /* returns number of weights adjusted */ ;
	long ROWLEN = rSes.rLearn->iOutputQty+1;
	/* clear all temp values BP0 */
	cuResetAllDeltasAndOutputs(rSes);
	/* re-evaluate sample to load temp values. BPI */
	cuEvalSingleSampleBeta(rSes, s, Net, (s==2), Signals, Zs, Wt, XInputs, YEval, dYEval);
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
		 /* divide the correction; delta = alpha * delta-star / n+1 (but alpha is always 1 for now). */
		Deltas[Net.iNeuronOfst[TOP]+i] = CxMultiplyRl( Deltastar, Net.dINV_S[TOP] );
	}
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
					++iReturn;
				}
			}
			// k index must begin at 1, neuron zero not valid for correction
			for (int k=1; k<iTributQTY; ++k) { /* contributions accumulated, now divide by dendrites+1. */
				Deltas[Net.iNeuronOfst[TRIB]+k] 
					= CxMultiplyRl( 
						Deltas[Net.iNeuronOfst[TRIB]+k] , 
						Net.dINV_S[TRIB] );
			}
		}
	}
	/* error distribution completed */
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
			++iReturn;
		}
	}
	/* re-evaluate sample to update temp values. */
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
					++iReturn;
				}
			}
			/* layer is complete. */
			cuEvalSingleSampleBeta(rSes, s, Net, false, Signals, Zs, Wt, XInputs, YEval, dYEval);
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
			++iReturn;
		}
	}
	/* backprop is complete. */

	return iReturn; /* number of weights updated. */
}


int TrainNNThresh(struct rohanContext& rSes, long bChangeWeights, int iSampleQty)
{mIDfunc 
/*! checks sampled outputs vs evaluated outputs, and returns number of samples that exceed threshold
 *  excessive samples are submitted for backpropagation if bChangeWeights is true.
 */
	int iReturn=0;
	double dDelta=0;
	long ROWLEN = rSes.rLearn->iOutputQty+1 ;
	if(iSampleQty<0 || iSampleQty>rSes.rLearn->lSampleQty){
		if (rSes.lSampleQtyReq<=0 || rSes.lSampleQtyReq>rSes.rLearn->lSampleQty)
			rSes.lSampleQtyReq=rSes.rLearn->lSampleQty;
		iSampleQty=rSes.lSampleQtyReq;
	}
	//adjust requested amount to available values
	for(long s=0; s<iSampleQty; ++s){  // loop over samples.
		int iOverMAX=0;
		for(int i=0; i<=rSes.rLearn->iOutputQty; ++i){  // loop over outputs.
			dDelta = (double) abs( rSes.rLearn->dDOutputs[ IDX2C( i, s, ROWLEN ) ] - rSes.rLearn->dYEval[ IDX2C( i, s, ROWLEN ) ] );
			 // printf("dDelta %f dDelta*2 %f, Sectors %d\n", dDelta, dDelta*2, Net.iSectorQty);
			if((dDelta*2)>rSes.rNet->iSectorQty)
				dDelta=rSes.rNet->iSectorQty-dDelta;
			 // printf("Sample %d, output %f eval %f delta %f\n", s, sam.dXInputs[rSes.rLearn->iInputQty+i], sam.dYEval[i], dDelta);
			if( dDelta > rSes.dMAX)  // if effective error exceeds MAX, make a note
				++iOverMAX;
			//if(gDevDebug){
			//	if(i==1)
			//		printf("[s%d %f > %f] %d \n", s, dDelta, rSes.dMAX, dDelta > rSes.dMAX);
			//}
		}
		if (iOverMAX!=0) {	 // if a note has been made. 
			++iReturn; // increment the number of excessive samples.
			if (bChangeWeights) {  // and correct weights if that is desired.
				cuBackpropSingleSample(rSes, s, *rSes.rNet, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rNet->Deltas, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval);
			}
		}
	}
	if(gDevDebug){
		cuDoubleComplex W = rSes.rNet->Wt[IDX2C( rSes.rNet->iWeightOfst[1]+1, 1, rSes.rNet->iDendrtQTY[1] )];
		printf("[%f + %fi]\n", W.x, W.y );
	}
	return (iReturn);
}


double RmseNN(struct rohanContext& rSes, long lSampleQtyReq)
{mIDfunc /*! checks sampled outputs vs evaluated outputs and calculates root mean squared error. */
	double dReturn=0.0;
	fprintf(rSes.hostBucket, "s\tD\t-\tY\t=\tArc\t\tErr\t\tSqrErr\t\taccum\n");
	// check if sample qty is outside the meaningful interval [1, all]
	if(lSampleQtyReq<=0 || lSampleQtyReq>rSes.rLearn->lSampleQty)
		lSampleQtyReq=rSes.rLearn->lSampleQty; // default to all if so
	for(long s=0; s<lSampleQtyReq; ++s){
		//struct rohanSample& sam = rSes.rLearn->rSample[s]; // loop over all requested samples and documented outputs
		for(int i=1; i<=rSes.rLearn->iOutputQty; ++i){
			double dDelta = (double)abs( rSes.rLearn->dDOutputs[IDX2C( i, s, (rSes.rLearn->iOutputQty+1))] - rSes.rLearn->dYEval[IDX2C( i, s, (rSes.rLearn->iOutputQty+1))] ); // delta = Desired - Yielded values
			fprintf(rSes.hostBucket, "%d\t%f\t%f\t%f\t", s, rSes.rLearn->dDOutputs[IDX2C( i, s, (rSes.rLearn->iOutputQty+1))], rSes.rLearn->dYEval[IDX2C( i, s, (rSes.rLearn->iOutputQty+1))], dDelta); 
			if(dDelta>(double)(rSes.rNet->iSectorQty/2)) 
				dDelta=((double)rSes.rNet->iSectorQty-dDelta); // set delta to the lesser arc length
			dReturn+=(dDelta*dDelta); // accumulate squared error 
			fprintf(rSes.hostBucket, "%f\t%f\t%f\n", dDelta, dDelta*dDelta, dReturn);
		}
	}
	fprintf(rSes.hostBucket, "RETURN: %f\t%f\n", dReturn, sqrt(dReturn/(double)(lSampleQtyReq*rSes.rLearn->iOutputQty)) );
	dReturn=sqrt(dReturn/(double)(lSampleQtyReq*rSes.rLearn->iOutputQty)); // take the root of the mean of the accumulated square error
	
	return rSes.dHostRMSE=dReturn;
}
