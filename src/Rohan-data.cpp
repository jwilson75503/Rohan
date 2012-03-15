/* Includes, cuda */
#include "stdafx.h"

#define _USE_MATH_DEFINES
#define ONE_PI 3.14159265358979323846264338327950288
#define TWO_PI 6.283185307179586476925286766558
//#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))
//#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))

extern int iDebugLvl, iTrace;
#include <boost/timer/timer.hpp>
//using boost::timer::cpu_timer;

int cuSectorTableMake(struct rohanContext &rSes)
{mIDfunc /// allocate and populate an array of complex coordinates for sectors on the unit circle of the complex plane
	double two_pi_div_sect_qty = TWO_PI/rSes.rNet->iSectorQty;

	rSes.rNet->cdcSectorBdry=(cuDoubleComplex*)malloc(rSes.rNet->iSectorQty * sizeof (cuDoubleComplex)); //point to array of cdc's
		mCheckMallocWorked(rSes.rNet->cdcSectorBdry)
	rSes.rNet->cdcAltSectorBdry=(cuDoubleComplex*)malloc(rSes.rNet->iSectorQty * sizeof (cuDoubleComplex)); //point to array of cdc's
		mCheckMallocWorked(rSes.rNet->cdcAltSectorBdry)
	for (int s=0; s<rSes.rNet->iSectorQty; ++s) {
		rSes.rNet->cdcSectorBdry[s].x=cos(s*two_pi_div_sect_qty);
		rSes.rNet->cdcSectorBdry[s].y=sin(s*two_pi_div_sect_qty);
		rSes.rNet->cdcAltSectorBdry[s]=cdcIdentity;
	}
	return rSes.rNet->iSectorQty;
}


long cuRandomizeWeights(struct rohanContext &rSes)
{mIDfunc /// generates random weights in [0..1]
	long lReturnValue=0;

	//for (int j=1; j < rSes.rNet->iLayerQty; ++j){ //no weights for layer 0
	struct rohanNetwork& Net = *rSes.rNet;
	for (int LAY=1; LAY<Net.iLayerQty; ++LAY){
		//struct rohanLayer& lay = rSes.rNet->rLayer[j];
		//for (int k=1; k <= lay.iNeuronQty; ++k){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no randomization for neuron 0
			//for (int i=0; i <= lay.iDendriteQty; ++i){
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				//cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				++lReturnValue;
				way.x=1.0-(double)(2*rand()/RAND_MAX); // range of values is -1.0 to +1.0
				way.y=1.0-(double)(2*rand()/RAND_MAX); // range of values is -1.0 to +1.0
			}
		}
	}
	printf("%d random weights on [0..1]\n",lReturnValue);
	//cuResetAllDeltasAndOutputs(rSes);
	return lReturnValue;
}


//long dualRandomizeWeights(struct rohanContext &rSes)
//{mIDfunc /// generates random weights in [0..1]
//	//cublasStatus csStatus;
//	long lReturnValue=0;
//
//	for (int j=1; j < rSes.rNet->iLayerQty; ++j){ //no weights for layer 0
//		struct rohanLayer& lay = rSes.rNet->rLayer[j];
//		// set neuron 0 's weights to nil
//		for (int i=1; i <= lay.iDendriteQty; ++i){
//			lay.Weights[IDX2C(i, 0, lay.iDendriteQty+1)].x=0;
//			lay.Weights[IDX2C(i, 0, lay.iDendriteQty+1)].y=0;
//		}
//		lay.Weights[IDX2C(0, 0, lay.iDendriteQty+1)].x=1; // NZero interior weight should always be equal to 1+0i
//		lay.Weights[IDX2C(0, 0, lay.iDendriteQty+1)].y=0;
//		for (int k=1; k <= lay.iNeuronQty; ++k){ // weights for neurons 1+
//			for (int i=0; i <= lay.iDendriteQty; ++i){
//				cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
//				way.x=(double)rand()/RAND_MAX;
//				way.y=(double)rand()/RAND_MAX;
//				++lReturnValue;
//			}
//		}
//		
//	}
//	printf("%d random weights on [0..1] generated and transferred.\n",lReturnValue);
//	cuResetAllDeltasAndOutputs(rSes);
//	////devResetAllDeltasAndOutputs(rSes);
//	return lReturnValue;
//}


cuDoubleComplex CxActivate(const cuDoubleComplex Z, struct rohanContext& rSes)
{/// applies ContActivation or discrete activation function to cx neuron output and returns Phi(Z)
	/// This fn should be phased out in favor of a GPU device vector based fn
	cuDoubleComplex phi;
	if (rSes.bContActivation) { // apply ContActivation activation function to weighted sum : phi(z)=z/|z|
		phi = CxDivideRl( Z, CxAbs( Z ) );
	}
	else {	// apply Discrete activation function to weighted sum : s=int(arctan(z)*k/2pi), phi(z)=(X(s),Y(s))
		double theta = atan2(Z.y, Z.x); // theta = arctan y/x
		int iSector = (int)((theta * rSes.rNet->dK_DIV_TWO_PI) + rSes.rNet->iSectorQty) % rSes.rNet->iSectorQty;
		//if (iSector<0)
		//	iSector+=rSes.rNet->iSectorQty;
		phi = rSes.rNet->cdcSectorBdry[iSector];
	}

	if(!_finite(phi.x) || !_finite(phi.y))
		printf("CxActivate: bad value from %f+%f !\n", phi.x, phi.y);

	return phi;
}


long cuConvertInputs(struct rohanContext& rSes, long lSample)
{mIDfunc /// converts sample inputs to complex NN input layer values //sam refs removed 11/6
	/// layer zero (inputs) is special
	/// virtual input neurons' outputs are network inputs converted to complex coords
	//FILE *fShow=rSes.debugHandle;
	long ROWLEN=rSes.rLearn->iInputQty+1;
	for (int i=0; i<ROWLEN; ++i){
		rSes.rNet->rLayer[0].ZOutputs[i]=rSes.rLearn->cdcXInputs[IDX2C( i, lSample, ROWLEN )];
	}
	return lSample;
}



long cuEvalMidTopLayers(struct rohanContext& rSes, long lSample)
{mIDfunc/// number crunches the middle and top layers of an MLMVN 
	//long s=lSample; //replace for-loop domain with requested sample index
	//int iLastLayer=rSes.rNet->iLayerQty-1;
	for (int L=1; L<rSes.rNet->iLayerQty; ++L){
		struct rohanLayer& lay = rSes.rNet->rLayer[L];
		int iLastNeuron=rSes.rNet->rLayer[L].iNeuronQty; // size of current layer
		int PL=L-1; // index of previous layer
		int iLastSignal=rSes.rNet->rLayer[L].iDendriteQty; // weight qty depends on size of previous layer
			cuDoubleComplex*& wt = rSes.rNet->rLayer[L].Weights;
			cuDoubleComplex*& oldOut = rSes.rNet->rLayer[PL].ZOutputs;
			cuDoubleComplex*& newOut = rSes.rNet->rLayer[L].ZOutputs;

		for (int k=0; k<=iLastNeuron; ++k){ //Neuron zero is not skipped, its output should be 1+0i as a check
			newOut[k]=cdcZero;
			
			for (int i=0; i<=iLastSignal; ++i){ //walk weights on inputs from previous layer
				newOut[k]=CxAddCx(newOut[k],CxMultiplyCx(wt[IDX2C( i, k, lay.iDendriteQty+1)], oldOut[i]));
			}

			// ACTIVATE //
			newOut[k]=CxActivate(newOut[k],rSes);
		}
	}
	return lSample;
}


long cuOutputConvert(struct rohanContext& rSes, long lSample)
{mIDfunc/// converts complex NN output layer values to evaluated sample outputs //sam refs removed 11/6
	long s=lSample; //replace for-loop domain with requested sample index
	int iLastLayer=rSes.rNet->iLayerQty-1;
	//struct rohanSample& sam = rSes.rLearn->rSample[s];
	long ROWLEN=rSes.rLearn->iOutputQty+1;
	struct rohanLayer& top = rSes.rNet->rLayer[iLastLayer];
	
	//printf("%dhos|", lSample);

	if (rSes.rLearn->iContOutputs){
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ // continuous conversion begins here 
			rSes.rLearn->dYEval[IDX2C( i, lSample, ROWLEN )]=FUnitCx(top.ZOutputs[i])*rSes.rNet->iSectorQty; // cx output is converted to angle [0,1), then multiplied by k, then stored with sample
			rSes.rLearn->cdcYEval[IDX2C( i, lSample, ROWLEN )]=top.ZOutputs[i]; // unactivated cx output is also stored with sample

			//printf("%g+%g\t",top.ZOutputs[i].x, top.ZOutputs[i].y);
		}
	}
	else{
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ // discrete conversion starts here
			rSes.rLearn->dYEval[IDX2C( i, lSample, ROWLEN )]=(double)floor(FUnitCx(top.ZOutputs[i])*rSes.rNet->iSectorQty); // cx output is converted to angle and mul;tiplied by k, but then the fraciton is dropped before storing
			rSes.rLearn->cdcYEval[IDX2C( i, lSample, ROWLEN )]=top.ZOutputs[i];
		}
	}

	//printf("\n");

	return lSample;
}

long cuConvertInputs16(struct rohanContext& rSes, long lSample, cuDoubleComplex * Sums)
{mIDfunc /// converts sample inputs to complex NN input layer values //sam refs removed 11/6
	/// layer zero (inputs) is special
	/// virtual input neurons' outputs are network inputs converted to complex coords
	//FILE *fShow=rSes.debugHandle;

	ShowMeResetSums(*rSes.rNet, Sums);
	//ShowMeDiffSums(*rSes.rNet, Sums, rSes.debugHandle, '>', 0,0,0);


	long ROWLEN=rSes.rLearn->iInputQty+1;
	for (int i=0; i<ROWLEN; ++i){
		rSes.rNet->rLayer[0].ZOutputs[i]=rSes.rLearn->cdcXInputs[IDX2C( i, lSample, ROWLEN )];


		Sums[i]=rSes.rLearn->cdcXInputs[IDX2C( i, lSample, ROWLEN )];
		ShowMeDiffSums(*rSes.rNet, Sums, stdout, '>', 0,i,0);
		//if(i==1)printf("cuCI  %d| %g+%g -> %g+%g\n", lSample, rSes.rNet->rLayer[0].ZOutputs[i].x, rSes.rNet->rLayer[0].ZOutputs[i].y, rSes.rLearn->cdcXInputs[IDX2C( i, lSample, ROWLEN )].x, rSes.rLearn->cdcXInputs[IDX2C( i, lSample, ROWLEN )].y);

	
	}
	return lSample;
}


long cuEvalMidTopLayers16(struct rohanContext& rSes, long lSample, cuDoubleComplex * Sums)
{mIDfunc/// number crunches the middle and top layers of an MLMVN 
	long s=lSample; //replace for-loop domain with requested sample index
	//int iLastLayer=rSes.rNet->iLayerQty-1;


	FILE *fShow = rSes.debugHandle;


	for (int L=1; L<rSes.rNet->iLayerQty; ++L){
		struct rohanLayer& lay = rSes.rNet->rLayer[L];
		int iLastNeuron=rSes.rNet->rLayer[L].iNeuronQty; // size of current layer
		int PL=L-1; // index of previous layer
		int iLastSignal=rSes.rNet->rLayer[L].iDendriteQty; // weight qty depends on size of previous layer
			cuDoubleComplex*& wt = rSes.rNet->rLayer[L].Weights;
			cuDoubleComplex*& oldOut = rSes.rNet->rLayer[PL].ZOutputs;
			cuDoubleComplex*& newOut = rSes.rNet->rLayer[L].ZOutputs;

		for (int k=0; k<=iLastNeuron; ++k){ //Neuron zero is not skipped, its output should be 1+0i as a check
			newOut[k]=cdcZero;
			Sums[rSes.rNet->iNeuronOfst[L]+k]=cdcZero;
			ShowMeDiffSums(*rSes.rNet, Sums, stdout, '|', L,k,0);

			
			for (int i=0; i<=iLastSignal; ++i){ //walk weights on inputs from previous layer
				newOut[k]=CxAddCx(newOut[k],CxMultiplyCx(wt[IDX2C( i, k, lay.iDendriteQty+1)], oldOut[i]));


				Sums[rSes.rNet->iNeuronOfst[L]+k]
					=CxAddCx(Sums[rSes.rNet->iNeuronOfst[L]+k],
						CxMultiplyCx( rSes.rNet->Wt[IDX2C(rSes.rNet->iWeightOfst[L]+k, i, rSes.rNet->iDendrtQTY[L])], Sums[rSes.rNet->iNeuronOfst[PL]+i]));
					ShowMeDiffSums(*rSes.rNet, Sums, stdout, ':', L,k,i);


			}

			// ACTIVATE //
			newOut[k]=CxActivate(newOut[k],rSes);

			
			Sums[rSes.rNet->iNeuronOfst[L]+k]
				=CxActivate(Sums[rSes.rNet->iNeuronOfst[L]+k], rSes);
				ShowMeDiffSums(*rSes.rNet, Sums, stdout, '/', L,k,0);


		}
	}
	return lSample;
}

long cuOutputConvert16(struct rohanContext& rSes, long lSample, cuDoubleComplex * Sums)
{mIDfunc/// converts complex NN output layer values to evaluated sample outputs //sam refs removed 11/6
	long s=lSample; //replace for-loop domain with requested sample index
	int iLastLayer=rSes.rNet->iLayerQty-1;
	//struct rohanSample& sam = rSes.rLearn->rSample[s];
	long ROWLEN=rSes.rLearn->iOutputQty+1;
	struct rohanLayer& top = rSes.rNet->rLayer[iLastLayer];
	
	//printf("%dhos|", lSample);

	if (rSes.rLearn->iContOutputs){
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ // continuous conversion begins here 
			rSes.rLearn->dYEval[IDX2C( i, lSample, ROWLEN )]=FUnitCx(top.ZOutputs[i])*rSes.rNet->iSectorQty; // cx output is converted to angle [0,1), then multiplied by k, then stored with sample
			rSes.rLearn->cdcYEval[IDX2C( i, lSample, ROWLEN )]=top.ZOutputs[i]; // unactivated cx output is also stored with sample

			//printf("%g+%g\t",top.ZOutputs[i].x, top.ZOutputs[i].y);
		}
	}
	else{
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ // discrete conversion starts here
			rSes.rLearn->dYEval[IDX2C( i, lSample, ROWLEN )]=(double)floor(FUnitCx(top.ZOutputs[i])*rSes.rNet->iSectorQty); // cx output is converted to angle and mul;tiplied by k, but then the fraciton is dropped before storing
			rSes.rLearn->cdcYEval[IDX2C( i, lSample, ROWLEN )]=top.ZOutputs[i];
		}
	}

	//printf("\n");

	return lSample;
}

//long devEvalNNLearnSet(struct rohanContext& rSes)
//{mIDfunc
///*! This will apply a MLMVN weight set to each sample of a learning set in turn and record the resulting final output for each.
// *  Discrete inputs and outputs are used. Real integers are convered via K-valued logic to complex coordinates,
// *  which are then product-summed by successive layers of neurons, then conveted back to integer output
// *  
// *  IMPORTANT: keep this code consistent with cuEvalSingleOutput in rohan-learn.cpp 
// */
//	long lSamplesEvaled=0;
//	// sample index, counts up
//	double two_pi_div_sect_qty = TWO_PI/rSes.rNet->iSectorQty;
//	if (rSes.lSampleQtyReq<=0 || rSes.lSampleQtyReq>rSes.rLearn->lSampleQty)
//		rSes.lSampleQtyReq=rSes.rLearn->lSampleQty;
//	// here beginneth ye main duty loop
//	for (long s=0; s<rSes.lSampleQtyReq; ++s){
//		struct rohanSample& sam = rSes.rLearn->rSample[s];
//		lSamplesEvaled+=devEvalSingleSample(rSes, s);
//		//lSamplesEvaled+=dualEvalSingleSample(rSes, s);
//		//lSamplesEvaled+=knlEvalSingleSample(rSes, rSes.rLearn, rSes.rNet, s);
//	// end of main duty loop, go back for the next sample
//	}
//	//ShowMeLS(rSes,false)+ShowMeWS(rSes,true);
//	return lSamplesEvaled; // return qty samples evaluated
//}


long cuEvalNNLearnSet(struct rohanContext& rSes)
{mIDfunc
/*! This will apply a MLMVN weight set to each sample of a learning set in turn and record the resulting final output for each.
 *  Discrete inputs and outputs are used. Real integers are convered via K-valued logic to complex coordinates,
 *  which are then product-summed by successive layers of neurons, then conveted back to integer output
 *  
 *  IMPORTANT: keep this code consistent with cuEvalSingleOutput in rohan-learn.cpp 
 */
	long lSamplesEvaled=0;
	// sample index, counts up

	double two_pi_div_sect_qty = TWO_PI/rSes.rNet->iSectorQty;
	if (rSes.lSampleQtyReq<=0 || rSes.lSampleQtyReq>rSes.rLearn->lSampleQty)
		rSes.lSampleQtyReq=rSes.rLearn->lSampleQty;
	// here beginneth ye main duty loop
	printf("HOST: Time to complete learn set eval:\n");
	{
		boost::timer::auto_cpu_timer t;
		for (long s=0; s<rSes.lSampleQtyReq; s+=1){
			//struct rohanSample& sam = rSes.rLearn->rSample[s];
			//lSamplesEvaled+=cuEvalSingleSample(rSes, s);
			lSamplesEvaled+=cuEvalSingleSampleBeta(rSes, s, *rSes.rNet, 0, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval); // fixed-length method working 2/14/12
		// end of main duty loop, go back for the next sample
		}
	}
	return lSamplesEvaled; // return qty samples evaluated
}


cuDoubleComplex ConvScalarCx(struct rohanContext& rSes, double Scalar)
{mIDfunc // converts a scalar value to a returned complex coordinate)

	cuDoubleComplex cdcReturn;
	
	if (Scalar > rSes.rNet->iSectorQty){
		cdcReturn.x=666.6;
		cdcReturn.y=666.6;
	}
	else {
		double theta=Scalar/rSes.rNet->dK_DIV_TWO_PI;
		cdcReturn.x=cos( theta);
		cdcReturn.y=sin( theta);			
	}

	return cdcReturn;
}