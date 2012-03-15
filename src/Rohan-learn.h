/* Includes, cuda */
#ifndef ROHAN_LEARN_H
#define ROHAN_LEARN_H
#include "Rohan.h"

////int devSetInputs(struct rohanContext& rSes, int iLayer);

////int devActivate(struct rohanContext& rSes, int iLayer);

////int devEvalSingleSample(struct rohanContext& rSes, long lSampleIdxReq);

int dualEvalSingleSample(struct rohanContext& rSes, long lSampleIdxReq)
;

//int cuEvalSingleSample(struct rohanContext& rSes, long lSampleIdxReq);

int cuEvalSingleSampleBeta(struct rohanContext& rSes, long s, rohanNetwork& Net, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;

int cuBackpropSingleSample(rohanContext& rSes, long lSampleIdxReq, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;

int cuEvalSingleOutput(rohanContext& rSes, long lSampleIdxReq, int iOutputIdxReq)
;

long OutputValidate(rohanContext& rSes)
;

////int devResetAllDeltasAndOutputs(struct rohanContext& rSes);

int cuResetAllDeltasAndOutputs(struct rohanContext& rSes)
;

////int devBackpropSingleSample(rohanContext& rSes, long lSampleIdxReq);

int dualBackpropSingleSample(struct rohanContext& rSes, long lSampleIdxReq)
;

int cuBackpropSingleSample(rohanContext& rSes, long lSampleIdxReq, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas)
;

int TrainNNThresh(struct rohanContext& rSes, long bChangeWeights)
;

double RmseNN(struct rohanContext& rSes, long lSampleQtyReq)
;

void cuCksum(struct rohanContext& rSes)
;

#endif