#ifndef COMPLEX_MATH_FUNC_H
#define COMPLEX_MATH_FUNC_H

cuDoubleComplex CxConjugate(cuDoubleComplex Z)
;

double CxAbs(cuDoubleComplex Z)
;

cuDoubleComplex CxMultiplyCx(cuDoubleComplex A, cuDoubleComplex B)
;

cuDoubleComplex CxMultiplyRl(cuDoubleComplex A, double Rl)
;

cuDoubleComplex CxInverse(cuDoubleComplex Z)
;

cuDoubleComplex CxDivideCx(cuDoubleComplex A, cuDoubleComplex B)
;

cuDoubleComplex CxDivideRl(cuDoubleComplex A, double Rl)
;

cuDoubleComplex CxSubtractCx(cuDoubleComplex A, cuDoubleComplex B)
;

cuDoubleComplex CxAddCx(cuDoubleComplex A, cuDoubleComplex B)
;

double FUnitCx(cuDoubleComplex A)
;

#endif
