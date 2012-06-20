/* Includes, cuda */
#include "stdafx.h"
#include <boost/timer/timer.hpp>

extern int gDebugLvl, gTrace;

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


long cuRandomizeWeightsBlock(struct rohanContext &rSes)
{mIDfunc /// generates random weights in [-1..0..1]
	long lReturnValue=0;

	struct rohanNetwork& Net = *rSes.rNet;
	for (int LAY=1; LAY<Net.iLayerQty; ++LAY){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no randomization for neuron 0
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				++lReturnValue;
				way.x=1.0-(double)(2*rand()/RAND_MAX); // range of values is -1.0 to +1.0
				way.y=1.0-(double)(2*rand()/RAND_MAX); // range of values is -1.0 to +1.0
			}
		}
	}
	printf("%d pseudo-random weights on [-1..0..1]\n",lReturnValue);
	
	return lReturnValue;
}


	
long cuRandomizeWeightsLayer(struct rohanContext &rSes)
{mIDfunc /// generates random weights in [-1..0..1]
	long lReturnValue=0;

	for (int j=1; j < rSes.rNet->iLayerQty; ++j){ //no weights for layer 0
		struct rohanLayer& lay = rSes.rNet->rLayer[j];
		for (int k=1; k <= lay.iNeuronQty; ++k){
			printf("\n[%d,%d] ",j,k);
			for (int i=0; i <= lay.iDendriteQty; ++i){
				cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
				++lReturnValue;
				way.x=1.0-(double)(2*rand()/RAND_MAX); // range of values is -1.0 to +1.0
				way.y=1.0-(double)(2*rand()/RAND_MAX); // range of values is -1.0 to +1.0
				printf("%d.", i);
			}
		}
	}
	printf("%d pseudo-random weights on [-1..0..1]\n",lReturnValue);

	return lReturnValue;
}


cuDoubleComplex CxActivate(const cuDoubleComplex Z, struct rohanNetwork& Net)
{/// applies ContActivation or discrete activation function to cx neuron output and returns Phi(Z)
	/// This fn should be phased out in favor of a GPU device vector based fn
	cuDoubleComplex phi;
	if (Net.bContActivation) { // apply ContActivation activation function to weighted sum : phi(z)=z/|z|
		phi = CxDivideRl( Z, CxAbs( Z ) );
	}
	else {	// apply Discrete activation function to weighted sum : s=int(arctan(z)*k/2pi), phi(z)=(X(s),Y(s))
		double theta = atan2(Z.y, Z.x); // theta = arctan y/x
		int iSector = (int)((theta * Net.dK_DIV_TWO_PI) + Net.iSectorQty) % Net.iSectorQty;
		phi = Net.cdcSectorBdry[iSector];
	}

	if(!_finite(phi.x) || !_finite(phi.y))
		printf("CxActivate: bad value from %f+%f !\n", phi.x, phi.y);

	return phi;
}


long cuEvalNNLearnSet(struct rohanContext& rSes, int iSampleQty)
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
	if (iSampleQty<=0 || iSampleQty>rSes.rLearn->lSampleQty){
		if (rSes.lSampleQtyReq<=0 || rSes.lSampleQtyReq>rSes.rLearn->lSampleQty)
			rSes.lSampleQtyReq=rSes.rLearn->lSampleQty;
		iSampleQty=rSes.lSampleQtyReq;
	}
	// here beginneth ye main duty loop
	{
		for (long s=0; s<iSampleQty; s+=1){
			lSamplesEvaled+=cuEvalSingleSampleBeta(rSes, s, *rSes.rNet, 0, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval); // fixed-length method working 2/14/12
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
