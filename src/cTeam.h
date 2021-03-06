#ifndef CTEAM_H
#define CTEAM_H

#include "Rohan.h"

#include "ShowMe.h"

//class cTeam /// for subclassing only, never instantiated in itself
//{
//		struct rohanContext * rSes;
//		struct rohanLearningSet * rLearn;
//		struct rohanNetwork * rNet;
//		class cBarge * Barge /*! The data-holding "object" currently in use. */;
//		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
//		long lIterationQty /*! Iterate this many times ofr sample set and stop. */;
//		double dTargetRMSE /*! Stop if RMSE falls to this level or below. */;
//	public:
//		long SetContext( struct rohanContext& rSes); // completed
//		long SetNetwork( struct rohanNetwork& rNet); // completed
//		long SetSamples( struct rohanLearningSet& rLearn); // completed
//		long SetDrover( class cDrover * cdDrover); // completed
//		long SetBarge( class cBarge * cbBarge); //completed
//		
//		virtual long LetHitch(struct rohanContext& rSes) /*! copy data to dev mem and attach structures to team */;
//		//long LetTaut(struct rohanContext& rSes) /*! update dev mem from host for epoch */;
//		//long LetSlack(struct rohanContext& rSes) /*! update host mem with results of epoch calculations */;
//		virtual long LetUnHitch(struct rohanContext& rSes) /*! release dev mem structures */;
//				
//		long GetTrainables( struct rohanContext& rSes, long lSampleQtyReq);
//		virtual double GetRmseNN( struct rohanContext& rSes, long lSampleQtyReq); // 
//		long GetEvalSingleSample( struct rohanContext& rSes, long lSampleIdxReq, char chMethod); /// implemented in cHostTeam, cDeviceTeam
//		long LetBackpropSingleSample( rohanContext& rSes, long lSampleIdxReq, char chMethod); /// implemented in cHostTeam, cDeviceTeam
//		long LetTrainNNThresh( rohanContext& rSes, long lSampleQtyReq, char chMethod); // completed
//		long SaveContext(int iMode);
//		long SaveWeights(int iMode);
//		long SaveEvaluation(int iMode);
//		long SetStopCriteria( long lIterationQty, double dTargetRMSE); /// specifies conditions for end of training task
//		long LetEvalSet( rohanContext& rS, long lSampleQtyReq, char chMethod); /// Submits a subset of the samples available forevaluation.
//	//private:
//};
//
//class cHostTeam: public cTeam // cTeam members available at their original access
//{/// A mule team, slow but sure; serial calculations run on the CPU
//	    static const char* const classname;
//	public:
//		cHostTeam( struct rohanContext& rSes){ SetContext(rSes); ShowMe(); const char* const classname = "cHostTeam"; } ; // end ctor
//		void ShowMe() /*! diagnostic identity display on screen */;
//		long GetEvalSingleSample( struct rohanContext& rSes, long lSampleIdxReq, char chMethod) /*! calculates NN outputs for a given sample with serial method */;
//		long LetBackpropSingleSample( rohanContext& rSes, long lSampleIdxReq, char chMethod) /*! procedure for training weights with MAX criterion */;
//		long LetTrainNNThresh( rohanContext& rSes, long lSampleQtyReq, char chMethod);
//};

class cDeviceTeam// revised to remove subclassing: public cTeam // cTeam members available at their original access
{/// A team of mighty stallions; parallel calculations in CUDA C running on the GPU
		struct rohanContext * rSes;
		struct rohanNetwork * rNet;
		struct rohanLearningSet * rLearn;
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
		long lIterationQty /*! Iterate this many times ofr sample set and stop. */;
		char bHitched;
		char bTaut;
			
	public:
		long SetContext( struct rohanContext& rSes); // completed
		long SetNetwork( struct rohanNetwork& rNet); // completed
		long SetSamples( struct rohanLearningSet& rLearn); // completed
		long SetDrover( class cDrover * cdDrover); // completed
		long SetBarge( class cBarge * cbBarge); //completed
		long GetTrainables( struct rohanContext& rSes, long lSampleQtyReq);
		//long GetEvalSingleSample( struct rohanContext& rSes, long lSampleIdxReq, char chMethod); /// implemented in cHostTeam, cDeviceTeam
		//long LetBackpropSingleSample( rohanContext& rSes, long lSampleIdxReq, char chMethod); /// implemented in cHostTeam, cDeviceTeam
		//long LetTrainNNThresh( rohanContext& rSes, long lSampleQtyReq, char chMethod); // completed
		long SaveContext(int iMode);
		long SaveWeights(int iMode);
		long SaveEvaluation(int iMode);
		long SetStopCriteria( long lIterationQty, double dTargetRMSE); /// specifies conditions for end of training task
		long LetEvalSet( rohanContext& rS, long lSampleQtyReq, char chMethod); /// Submits a subset of the samples available forevaluation.
	cDeviceTeam( struct rohanContext& rSes);//, struct rohanNetwork& rNet, rohanLearningSet& rLearn);///{ SetContext(rSes); ShowMe(); } ; /// end ctor
		//cDeviceTeam( struct rohanContext& rSes); /// ctor
		void ShowMe() /*! diagnostic identity display on screen */;
		long LetHitch(struct rohanContext& rSes) /*! copy data to dev mem and attach structures to team */;
			long TransferContext(struct rohanContext& rSes, char Direction) /*! copy rSes 0D members to dev mem */;
			long CopyNet(struct rohanContext& rSes, char Direction) /*! copy rNet members to dev mem */;
				long TransferNet(struct rohanContext& rSes, char Direction) /*! copy Net params to dev mem */;
			long CopyLearnSet(struct rohanContext& rSes, char Direction) /*! copy rLearn members to dev mem */;
		double GetRmseNN( struct rohanContext& rSes, long lSampleQtyReq) /*! checks sampled outputs vs evaluated outputs and calculates root mean squared error. */ ;
			long LetTaut(struct rohanContext& rSes) /*! update dev mem from host for epoch */;
				long TransferLayers(struct rohanContext& rSes, char Direction);
			long LetSlack(struct rohanContext& rSes) /*! update host mem with results of epoch calculations */;
				long TransferOutputs(struct rohanContext& rSes, char Direction);
		long LetUnHitch(struct rohanContext& rSes) /*! release dev mem structures */;
		
		long GetEvalSingleSample( struct rohanContext& rSes, long lSampleIdxReq, char chMethod) /*! calculates NN outputs for a given sample with GPU method */;
		long LetBackpropSingleSample( rohanContext& rSes, long lSampleIdxReq, int o, char chMethod) /*! procedure for training weights with MAX criterion */;
		long LetTrainNNThresh( rohanContext& rSes, long lSampleQtyReq, int o, char chMethod, double dTargetRMSE, int iEpochLength);
		char GetHitched();
		char GetTaut();
		int CUDAverify(struct rohanContext& rSes);
};

//class cDualTeam: public cTeam // cTeam members available at their original access
//{/// A twin "adapter" team, for comparing outputs of host and device calculations
//		class cHostTeam * CuTeam;
//		class cDeviceTeam * DevTeam;
//	    static const char* const classname;
//	public:
//		cDualTeam( struct rohanContext& rSes){ SetContext(rSes); ShowMe(); const char* const classname = "cDualTeam"; } ; // end ctor
//		void ShowMe() /*! diagnostic identity display on screen */;
//		long GetEvalSingleSample( struct rohanContext& rSes, long lSampleIdxReq, char chMethod) /*! calculates NN outputs for a given sample with GPU method */;
//		long LetBackpropSingleSample( rohanContext& rSes, long lSampleIdxReq, char chMethod) /*! procedure for training weights with MAX criterion */;
//		long LetTrainNNThresh( rohanContext& rSes, long lSampleQtyReq, char chMethod);
//};

#endif