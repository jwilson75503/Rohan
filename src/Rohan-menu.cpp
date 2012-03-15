/* Includes, cuda */

#include <iostream>
using namespace std;
#include "Rohan-data.h"
#include "Rohan-learn.h"
#include "ShowMe.h"
#include <conio.h> //for _getch

extern int iDebugLvl, iWarnings, iErrors, iTrace;
extern long bCUDAavailable;
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))


