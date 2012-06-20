/* Includes, cuda */
#include "stdafx.h"


cuDoubleComplex CxConjugate(const cuDoubleComplex Z)
{/// returns complex conjugate of Z
	cuDoubleComplex C;

	C.x = Z.x;
	C.y = - Z.y;
	
	return C;
}

double CxAbs(const cuDoubleComplex Z)
{/// returns absolute value of Z; aka modulus or magnitude
	double abs;
	
	abs = sqrt(Z.x * Z.x + Z.y * Z.y);

	return abs;
}

cuDoubleComplex CxMultiplyCx(const cuDoubleComplex A, const cuDoubleComplex B)
{/// returns product of complex factors A and B
	cuDoubleComplex C;

	// FOIL procedure for binomial multiplication: first, outside, inside, last
	// first is real times real, last is im times im
	C.x = A.x * B.x - A.y * B.y;
	// outside and inside are both real times imaginary
	C.y = A.x * B.y + A.y * B.x;

	return C;
}

cuDoubleComplex CxMultiplyRl(const cuDoubleComplex A, const double Rl)
{/// returns product of complex factor A and real factor Rl
	cuDoubleComplex B;

	B.x = Rl * A.x;
	B.y = Rl * A.y;
	
	return B;
}

cuDoubleComplex CxInverse(const cuDoubleComplex Z)
{/// returns invers of complex value Z
	cuDoubleComplex C; double recip_denom;

	// inverse of z = z-bar / square of abs(z)
	recip_denom = 1 / (Z.x * Z.x + Z.y * Z.y);
	C.x = Z.x * recip_denom;
	C.y = -Z.y * recip_denom; // C is now inverse of Z

	return C;
}

cuDoubleComplex CxDivideCx(const cuDoubleComplex A, const cuDoubleComplex B)
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

cuDoubleComplex CxDivideRl(const cuDoubleComplex A, const double Rl)
{/// returns quotient of complex dividend A and real divisor Rl
	cuDoubleComplex B;
	double recip_Rl;

	recip_Rl=1/Rl;
	B.x = A.x * recip_Rl; 
	B.y = A.y * recip_Rl;
	
	return B;
}

cuDoubleComplex CxSubtractCx(const cuDoubleComplex A, const cuDoubleComplex B)
{/// returns difference of complex subends A and B
	cuDoubleComplex C;

	C.x = A.x - B.x;
	C.y = A.y - B.y;
	
	return C;
}

cuDoubleComplex CxAddCx(const cuDoubleComplex A, const cuDoubleComplex B)
{/// returns the sum of complex addends A and B
	cuDoubleComplex C;

	C.x = A.x + B.x;
	C.y = A.y + B.y;
	
	return C;
}

double FUnitCx(const cuDoubleComplex A)
{/// returns the unitary angle of A in non-negative values [0,1)
	double fUnit;
	
	fUnit=atan2(A.y,A.x);
	fUnit*=ONE_OVER_TWO_PI;
	if(fUnit<0)
		++fUnit;

	if(fUnit!=fUnit)
		printf("FUnitCX: bad value from %f+%f !\n", A.x, A.y);

	return fUnit;
}
