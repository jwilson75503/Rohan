#ifndef CDROVER_H
#define CDROVER_H


class cDrover
{/// Provisional name for Controller/UI handler.
		struct rohanContext * rSes;
	private:
        static const char* const classname;
		class cDeviceTeam * Team /*! The calculating "engine" currently in use. */;
	public:
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		cDrover( rohanContext& rC, rohanLearningSet& rL, rohanNetwork& rN, cBarge& cB, cDeviceTeam& cdT){ SetContext(rC, rL, rN); SetDroverBargeAndTeam(cB, cdT); /*ShowMe();*/ }; // end ctor
			void ShowMe();
			long SetContext( struct rohanContext& rC, struct rohanLearningSet& rL, struct rohanNetwork& rN); // completed
			long SetDroverBargeAndTeam( class cBarge& cbB, class cDeviceTeam& cdtT); // completed
		long DoAnteLoop(int argc, char * argv[]); /// prepares all parameters and data structures necesary for learning and evaluation.
			int SetProgOptions(struct rohanContext& rSes, int argc, char * argv[]); // interprets command line options
			long ObtainGlobalSettings(struct rohanContext& rSes); /// sets initial and default value for globals and settings
			long AskSampleSetName(struct rohanContext& rSes) ;  /// chooses the learning set to be worked with Ante-Loop
			long ShowDiagnostics(struct rohanContext& rSes, struct rohanNetwork& rNet); /// show some statistics, dump weights, and display warning and error counts
				double cDrover::RmseEvaluateTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iSampleQty); /// runs tests for RMSE and evaluation on both host and GPU
				int cDrover::ClassifyTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iSampleQty); /// runs classification tests on both host and GPU
				double cDrover::BackPropTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iThreads, int iSampleQty); /// runs tests for backward propagation on both host and GPU
		long DoMainLoop(struct rohanContext& rSes); /// Trains a weight set to more closely reproduce the sampled outputs from the sampled inputs, and other options.
			int DisplayMenu(int iMenuNum, struct rohanContext& rSes);
			int CLIbase(struct rohanContext& rSes);
			int cDrover::GetWeightSet(struct rohanContext& rSes);
			long LetInteractiveEvaluation(struct rohanContext& rSes); 
			long LetInteractiveLearning(struct rohanContext& rSes);
			long LetUtilities(struct rohanContext& rSes);
		long DoPostLoop(struct rohanContext& rSes); /// Final operations including freeing of dynamically allocated memory are called from here. 
			long DoEndItAll(struct rohanContext& rSes); /// prepares for graceful ending of program
		int GetNNTop(struct rohanContext& rSes);
		void RLog(struct rohanContext& rSes, char * sLogEntry);
};

//int AnteLoop(struct rohanContext& rSes, int argc, char * argv[]);
//int GetGlobalSettings(struct rohanContext& rSes);
int BeginSession(struct rohanContext& rSes);

int GetWeightSet(struct rohanContext& rSes);
int GetSampleSet(struct rohanContext& rSes);
int ReGetSampleSet(struct rohanContext& rSes);
int PrepareNetwork(struct rohanContext& rSes);
int DirectoryEnsure(char * sPath);
int GetUserDocPath(char * sPath);


//int InteractiveEvaluation(struct rohanContext& rSes);
//int InteractiveLearning(struct rohanContext& rSes);
//void PostLoop(struct rohanContext& rSes);


#endif
