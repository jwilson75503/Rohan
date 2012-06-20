#ifndef ROHAN_KERNEL_H
#define ROHAN_KERNEL_H

extern"C"
int SetDevDebug(int gDevDebug);

extern"C"
int knlBackProp(struct rohanContext& rSes, long lSampleQtyReq, long o, char Option, int iBlocks, int iThreads);

__global__ void mtkBackPropMT( long lSampleQtyReq, long o, char Option);

__device__ void subkBackPropRoptMT(long lSampleQtyReq, long o);

__device__ void subkBackPropMT(long lSample, long o);

__device__ void subkBackPropSoptMThread(long s, int o, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval );

__device__ void subkBackPropSoptMWarp(long s, int o, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval );

__device__ void subkBackPropEoptMT(long lSampleQtyReq, long o);

extern"C"
double knlFFeRmseOpt(struct rohanContext& rSes, long lSampleQtyReq, long o, char Option, int iBlocks, int iThreads);

__global__ void mtkFFeRmseOptMT( long lSampleQtyReq, long o, char Option);

__device__ void subkRmseMT(long lSampleQtyReq, long o, int OUTROWLEN, double * dSqrErr);

__device__ void subkRmseMTBeta(long lSampleQtyReq, long o, int OUTROWLEN, double * dSqrErr);

__device__ void subkEvalSampleBetaMT(rohanContext& Ses, long s, rohanNetwork& Net, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval, double * dSqrErr);

__device__ void subkEvalSampleSingleThread(long s, char Option, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval, double * dSqrErr);

__device__ void subkShowMeDiffSums( cuDoubleComplex * Sums, char cSymbol, int x1, int x2, int x3);

__device__ void subkShowMeResetSums( cuDoubleComplex * Sums);

__device__ void subkEvalSingleSampleUT(long lSample);

__device__ void subkConvertInputsUT( long lSample);

__device__ void subkEvalMidTopLayersUT( long lSample);

__device__ void subkOutputConvertUT(long lSample);

__device__ double FUnitCxUT(const cuDoubleComplex A);

__device__ cuDoubleComplex CxAddCxUT(const cuDoubleComplex A, const cuDoubleComplex B);

__device__ cuDoubleComplex CxMultiplyCxUT(const cuDoubleComplex A, const cuDoubleComplex B);

__device__ cuDoubleComplex CxActivateUT(const cuDoubleComplex Z, rohanNetwork& Net);

__device__ cuDoubleComplex CxMultiplyRlUT(const cuDoubleComplex A, const double Rl);

__device__ cuDoubleComplex CxDivideRlUT(const cuDoubleComplex A, const double Rl);

__device__ double CxAbsUT(const cuDoubleComplex Z);

__device__ cuDoubleComplex CxSubtractCxUT(const cuDoubleComplex A, const cuDoubleComplex B);

__device__ cuDoubleComplex CxDivideCxUT(const cuDoubleComplex A, const cuDoubleComplex B);

__device__ cuDoubleComplex CxConjugateUT(const cuDoubleComplex Z);

//__device__ long d_answer;

extern "C" 
long knlCRC32Buf(char * buffer, unsigned int length);

__global__ __device__ void mtkCRC32Buf(char * buffer, unsigned int length);

__device__ long subkCrc32buf(char *buf, size_t len);

__device__ double atomicAdd(double* address, double val);

__device__ void __checksum(char * sLabel);

#endif
