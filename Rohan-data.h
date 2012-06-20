/* Includes, cuda */


int cuMakeNNStructures(struct rohanContext& rSes)
;
int cuSectorTableMake(struct rohanContext& rSes)
;
long cuRandomizeWeightsBlock(struct rohanContext& rSes)
;
long cuRandomizeWeightsLayer(struct rohanContext& rSes)
;
cuDoubleComplex CxActivate(const cuDoubleComplex Z, struct rohanNetwork& Net)
;
long cuConvertInputs16(struct rohanContext& rSes, long lSample, cuDoubleComplex * Sums)
;
long cuEvalMidTopLayers16(struct rohanContext& rSes, long lSample, cuDoubleComplex * Sums)
;
long cuOutputConvert16(struct rohanContext& rSes, long lSample, cuDoubleComplex * Sums)
;
long cuConvertInputs(struct rohanContext& rSes, long lSample)
;
long cuEvalMidTopLayers(struct rohanContext& rSes, long lSample)
;
long cuOutputConvert(struct rohanContext& rSes, long lSample)
;
long cuEvalNNLearnSet(struct rohanContext& rSes, int iSampleQty)
;
////int cuFreeNNTop(struct rohanContext& rSes);
int cuFreeLearnSet(struct rohanContext& rSes)
;
int cuFree(struct rohanContext& rSes)
;


int devCopyNNStructures(struct rohanContext& rSes)
;
int devCopySectorTable(struct rohanContext& rSes)
;
////long devOutputConvert(struct rohanContext& rSes, long lSample);
////long devEvalNNLearnSet(struct rohanContext& rSes);
////long dualRandomizeWeights(struct rohanContext& rSes);

cuDoubleComplex ConvScalarCx(struct rohanContext& rSes, double Scalar)
;
