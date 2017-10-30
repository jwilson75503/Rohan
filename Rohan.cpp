/*! \mainpage Rohan Multivalued Neural Network Simulator
 *
 * \section intro_sec Introduction
 *
 * Rohan was developed by Jeff Wilson <jwilson@clueland.com> at the Texas A&M University - Texarkana Computational Intelligence Laboratory < http://www.tamut.edu/CIL/ > under the direction of Dr Igor Aizenberg.
 *
 * Funded by National Science Foundation grant #0925080
 *
 * \section install_sec Installation
 *
 * \subsection step1 Step 1: Opening the box
 *  
 * Hopefully included in the 1.0 or later release.
 */
// Rohan.cpp : Defines the entry point for the console application.

/* Includes, cuda */
#include "stdafx.h"

/// globals
int gDebugLvl=0, gDevDebug=0, gTrace=0;
float gElapsedTime=0.0, gKernelTimeTally=0.0;


int _tmain(int argc, _TCHAR* argv[])
{mIDfunc/// general program procedure is to setup preparations for the duty loop, execute it, then do housekeeping after
	struct rohanContext rSes /* This is the master session context object, with pointers to the learning set and network objects as members */;
	struct rohanNetwork rNet;
	struct rohanLearningSet rLearn;

	// create class objects
 
	cDeviceTeam cdtHorse(rSes); // the horse handles GPU computation kernels and their currency
	cBarge cbBarge(rSes); // the barge loads and holds common data like the learning set and weights
	cDrover cdDrover(rSes, rLearn, rNet, cbBarge, cdtHorse); // the drover handles the user interface and directs the other objects

	// proceed with operations based on session variables and external settings

 	if(cdDrover.DoAnteLoop(argc, argv)){
		cdDrover.DoMainLoop(rSes);
	}
 
	cdDrover.DoPostLoop(rSes);
	
	// end of operations
	exit (0);
}
