#ifndef ROHAN_CLASS_H
#define ROHAN_CLASS_H

#include "Rohan.h"
#include "ShowMe.h"
#include <iostream>

class cBarge
{/// Represents the load of data upon which the computational work is to be performed.
		struct rohanContext * rSes;
		struct rohanLearningSet * rLearn;
		struct rohanNetwork * rNet;
		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
		class cDeviceTeam * Team /*! The calculating "engine" currently in use. */;
	private:
        static const char* const classname;
	public:
		cBarge( struct rohanContext& rSes){ SetContext(rSes); /*ShowMe();*/ } ; // end ctor
		void ShowMe();
		long SetContext( struct rohanContext& rSes); // completed
			long SetDrover( class cDrover * cdDrover); // completed
			long SetTeam( class cDeviceTeam * cdtTeam); // completed
		long ObtainSampleSet(struct rohanContext& rSes); /// chooses and loads the learning set to be worked with Ante-Loop
			long DoLoadSampleSet(struct rohanContext& rSes, FILE *fileInput); /// pulls in values from .txt files, used for testing before main loop
			long CurateSectorValue(struct rohanContext& rSes); /// compares sector qty to sample values for adequate magnitude
			long CompleteHostLearningSet(struct rohanContext& rSes); /// allocate, fill cx converted value & alt values, all in host memory
				//long cuDoubleComplex ConvScalarCx(struct rohanContext& rSes, long Scalar); // converts a scalar value to a returned complex coordinate)
			//long LetCplxCopySamples(struct rohanContext& rSes); //load complex samples into the parallel structures in the host memory
		long DoPrepareNetwork(struct rohanContext& rSes); /// sets up network poperties and data structures for use
			long DoCuPrepareNetwork(struct rohanContext& rSes); /// sets up network properties and data structures for use in host memory space
		long LetWriteWeights(struct rohanContext& rSes); /// saves weigth values to disk
		long ShowDiagnostics();
		long DoCuFree(struct rohanContext &rSes);
			long cuFreeNNTop(struct rohanContext &rSes); /// frees data structures related to network topology
			long cuFreeLearnSet(struct rohanContext &rSes); /// free the learning set of samples
};

class cDrover
{/// Provisional name for Controller/UI handler.
		struct rohanContext * rSes;
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		class cDeviceTeam * Team /*! The calculating "engine" currently in use. */;
	private:
        static const char* const classname;
	public:
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