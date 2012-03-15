#ifndef CDROVER_H
#define CDROVER_H

#include "Rohan.h"
#include "ShowMe.h"
#include <iostream>

class cDrover
{/// Provisional name for Controller/UI handler.
		struct rohanContext * rSes;
	private:
        static const char* const classname;
		class cDeviceTeam * Team /*! The calculating "engine" currently in use. */;
	public:
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		
		cDrover( struct rohanContext& rSes){ SetContext(rSes); /*ShowMe();*/ } ; // end ctor
			void ShowMe();
			long SetContext( struct rohanContext& rSes); // completed
		long DoAnteLoop(int argc, char * argv[],  class cDrover * cdDrover, class cBarge * cbBarge, class cDeviceTeam * cdtTeam); /// preps all params, contexts, amd data structures necesary for learning and evaluation.
			long SetDroverBargeAndTeam( class cDrover * cdDrover, class cBarge * cbBarge, class cDeviceTeam * cdtTeam); // completed
			long ObtainGlobalSettings(struct rohanContext& rSes); /// sets initial and default value for globals and settings
			long AskSampleSetName(struct rohanContext& rSes) ;  /// chooses the learning set to be worked with Ante-Loop
			long ShowDiagnostics(struct rohanContext& rSes); /// show some statistics, dump weights, and display warning and error counts
		long DoMainLoop(struct rohanContext& rSes); /// Trains a weight set to more closely reproduce the sampled outputs from the sampled inputs, and other options.
			long LetInteractiveEvaluation(struct rohanContext& rSes); 
			long LetInteractiveLearning(struct rohanContext& rSes);
		long DoPostLoop(struct rohanContext& rSes); /// Final operations including freeing of dynamically allocated memory are called from here. 
			long DoEndItAll(struct rohanContext& rSes); /// prepares for graceful ending of program
};

//int AnteLoop(struct rohanContext& rSes, int argc, char * argv[]);
//int GetGlobalSettings(struct rohanContext& rSes);
int BeginSession(struct rohanContext& rSes);
int GetNNTop(struct rohanContext& rSes);
	long cuFreeNNTop(struct rohanContext &rSes);
int GetWeightSet(struct rohanContext& rSes);
int GetSampleSet(struct rohanContext& rSes);
int ReGetSampleSet(struct rohanContext& rSes);
int PrepareNetwork(struct rohanContext& rSes);
//void MainLoop(struct rohanContext& rSes);
int SilentDiagnostics(struct rohanContext& rSes);
//int InteractiveEvaluation(struct rohanContext& rSes);
//int InteractiveLearning(struct rohanContext& rSes);
//void PostLoop(struct rohanContext& rSes);


#endif
