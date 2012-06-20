/* Includes, cuda */
#ifndef ROHAN_LEARN_H
#define ROHAN_LEARN_H

////int devSetInputs(struct rohanContext& rSes, int iLayer);

////int devActivate(struct rohanContext& rSes, int iLayer);

////int devEvalSingleSample(struct rohanContext& rSes, long lSampleIdxReq);

int dualEvalSingleSample(struct rohanContext& rSes, long lSampleIdxReq)
;

int cuEvalSingleSampleBeta(struct rohanContext& Ses, long s, rohanNetwork& Net, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
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

int cuBackpropLearnSet(rohanContext& rSes, long s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;

int cuBackpropSingleSample(rohanContext& rSes, long s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;

int TrainNNThresh(struct rohanContext& rSes, long bChangeWeights, int iSampleQty)
;

double RmseNN(struct rohanContext& rSes, long lSampleQtyReq)
;

void cuCksum(struct rohanContext& rSes)
;

#endif