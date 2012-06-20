#include "stdafx.h"

extern int gDebugLvl, gDevDebug, gTrace;
extern long bCUDAavailable;

//////////////// class cDeviceTeam begins ////////////////

cDeviceTeam::cDeviceTeam( struct rohanContext& rSes)//, struct rohanNetwork& rNet, rohanLearningSet& rLearn)
{mIDfunc
	SetContext(rSes); // attach to host-side data structures
	SetNetwork(*rSes.rNet);
	SetSamples(*rSes.rLearn);

	// informally acknowledge class
	//ShowMe();
	//const char* const classname = "cDeviceTeam";
	
	bHitched=FALSE;
	bTaut=FALSE;
	// end ctor
}

char cDeviceTeam::GetHitched()
{	
	return bHitched;
}

char cDeviceTeam::GetTaut()
{	
	return bTaut;
}


long cDeviceTeam::SetContext( rohanContext& rC)
{mIDfunc/// enables pointer access to master context struct
	rSes = &rC;
	return 0;
}

long cDeviceTeam::SetNetwork( rohanNetwork& rN)
{mIDfunc/// enables pointer access to weight sets and layer sizes, etc
	rNet = &rN;
	return 0;
}

long cDeviceTeam::SetSamples( rohanLearningSet& rL)
{mIDfunc/// enables pointer access to master sample struct
	rLearn = &rL;
	return 0;
}

long cDeviceTeam::SetBarge( class cBarge * cbBarge)
{mIDfunc/// enables pointer access to active Barge object
	Barge = cbBarge;
	return 0;
}

long cDeviceTeam::SetDrover( class cDrover * cdDrover)
{mIDfunc/// enables pointer access to active Drover object
	Drover = cdDrover;
	return 0;
}


void cDeviceTeam::ShowMe()
{mIDfunc
	//ShowMeSes(* rSes, false);
	printf("I'm a mighty stallion!\n");
}

long cDeviceTeam::LetEvalSet( rohanContext& rSes, long lSampleQtyReq, char chMethod)
{mIDfunc/// Submits a subset of the samples available forevaluation.
	/// Size of the subet is controlled by lSampleQtyReq
	long lFinalSample;
	// check request is not too large or too small
	if (0<lSampleQtyReq && lSampleQtyReq<rSes.rLearn->lSampleQty)
		lFinalSample=lSampleQtyReq;
	else
		lFinalSample=rSes.rLearn->lSampleQty;
	// choose host submission loop or device
	if(chMethod=='H' || chMethod=='h'){
		for (long i=0; i<lFinalSample; ++i){
			cuEvalSingleSampleBeta(rSes, i, *rSes.rNet, 0, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval);
		}
	}
	else {// D for GPU device XX
		if(!bTaut){
			LetTaut(rSes);
			knlFFeRmseOpt( rSes, lFinalSample, 1, 'R', rSes.iEvalBlocks, rSes.iEvalThreads);
			LetSlack(rSes);
		} else
		knlFFeRmseOpt( rSes, lFinalSample, 1, 'R', rSes.iEvalBlocks, rSes.iEvalThreads);
	}
	return lFinalSample;
}

long cDeviceTeam::LetTrainNNThresh( rohanContext& rSes, long lSampleQtyReq, int o, char chMethod, double dTargetRMSE, int iEpochLength)
{mIDfunc/// Submits a subset of the samples available for backprop learning.
/// Size of the subet is controlled by lSampleQtyReq, o indictaes which output(s)
//
// chMethod controls option used:
// Option S - single sample correction only XX unimplemented?
// Option E - keep existing weights, count trainable samples only
// Option R - perform corrections for all trainable samples
	bool bInternalTaut=false;
	long lReturn, lFinalSample, count=1;
	double RMSE; //=dTargetRMSE;
	// check request is not too large or too small
	if (0<lSampleQtyReq && lSampleQtyReq<rSes.rLearn->lSampleQty)
		lFinalSample=lSampleQtyReq;
	else
		lFinalSample=rSes.rLearn->lSampleQty;
	
	if(!bTaut){
		LetTaut(rSes);
		bInternalTaut=true;
	}
	RMSE=knlFFeRmseOpt( rSes, lFinalSample, 1, 'R', rSes.iEvalBlocks, rSes.iEvalThreads); // check RMSE
	lReturn=knlBackProp( rSes, lFinalSample, o, chMethod, rSes.iBpropBlocks, rSes.iBpropThreads); // do training

	if(chMethod=='R') {
		RMSE=knlFFeRmseOpt( rSes, lFinalSample, 1, 'R', rSes.iEvalBlocks, rSes.iEvalThreads); // check RMSE
		while(dTargetRMSE<RMSE && lReturn && count < iEpochLength && chMethod=='R'){ // target not met, trainable samples left
			//if(gDevDebug)conPrintf(">>DEVICE: RMSE %f, %d trained, count %d of %d\n", RMSE, lReturn, count, iEpochLength);
			lReturn=knlBackProp( rSes, lFinalSample, o, chMethod, rSes.iBpropBlocks, rSes.iBpropThreads); // do more training
			++count; // increment counter
			RMSE=knlFFeRmseOpt( rSes, lFinalSample, 1, 'R', rSes.iEvalBlocks, rSes.iEvalThreads); // check RMSE anew
		}
		if(bInternalTaut)
			LetSlack(rSes);
		//if(gDevDebug)conPrintf(">>DEVICE: RMSE %f, %d trained, count %d of %d\n", RMSE, lReturn, count, iEpochLength);
		rSes.dDevRMSE=RMSE; // update device-achieved RMSE
		//if(gDevDebug)conPrintf(">>DEVICE: Epoch ended with %d iterations.\n", count);
		//if(dTargetRMSE>=RMSE){
		//	conPrintf(">>DEVICE:  Achieved RMSE target %f.\n", dTargetRMSE);
			//rSes.dTargetRMSE=RMSE*0.9; // always want to decrease RMSE by 10%
			//conPrintf(" New target is %f.\n", rSes.dTargetRMSE);
		//}
	}
	else{
		if(bInternalTaut)
			LetSlack(rSes);
	}	

	return lReturn;
}

long cDeviceTeam::LetBackpropSingleSample( rohanContext& rSes, long lSampleIdxReq, int o, char chMethod)
{mIDfunc
	long lReturn=knlBackProp( rSes, lSampleIdxReq, o, 'S', rSes.iBpropBlocks, rSes.iBpropThreads); // S indicates single bprop operation

	return lReturn;
}

double cDeviceTeam::GetRmseNN(struct rohanContext& rSes, long lSampleQtyReq)
{mIDfunc/*! checks sampled outputs vs evaluated outputs and calculates root mean squared error. */
//Option will determine if existing data is used (E, not implemented) or refreshed (R) XX
// param 3 is output index		
	return knlFFeRmseOpt( rSes, lSampleQtyReq, 1, 'R', rSes.iEvalBlocks, rSes.iEvalThreads);
}

long cDeviceTeam::LetHitch(struct rohanContext& rSes)    
{mIDfunc/*! \callgraph \callergraph copy data to device memory space and attach structures to Team */
if (bHitched){
		printf("cDeviceTeam already hitched.\n");
		return FALSE;
	}
	else{
		//printf("cDeviceTeam begin hitch process.\n");

		// BEGIN context data structure members
 		TransferContext(rSes, 'D');
		
		/*! BEGIN network architecture structure members */
		CopyNet( rSes, 'D' );
		
		/*! BEGIN learning set structure members */
		CopyLearnSet(rSes, 'D');
		
		/// end members and structures
		if(rSes.bConsoleUsed)printf("H+T-->\n");
		bHitched=TRUE; bTaut=TRUE;
		return TRUE;
	}
}


long cDeviceTeam::TransferContext(struct rohanContext& rSes, char Direction)
{mIDfunc/*! copy rSes and members to dev mem */
	
	long SIZE = sizeof(rSes);
	
	if(Direction=='D'||Direction=='d') {
		// publish Context structure to device
			mCheckCudaWorked
		cudaMemcpyToSymbol( "devSes", &rSes, sizeof(rSes) );
			mCheckCudaWorked
	}
	else {
		// retrieve rSes from device
		cudaMemcpyFromSymbol( &rSes, "devSes", sizeof(rSes) );
			mCheckCudaWorked
	}
	if(crc32buf( (char*)&rSes, SIZE )!=knlCRC32Buf( (char*)rSes.devSes, SIZE ))
		printf("FAILURE copying context\n");
	//else
	//	printf("%c context faithfully copied\n", Direction);
	
	return 0;
}

long cDeviceTeam::CopyNet(struct rohanContext& rSes, char Direction)
{mIDfunc/*! copy NN arch and layer structures, their contents, and member values to dev memory */
// layer structures removed 3/31/12 JAW
	cudaMemcpyKind kind; //cuDoubleComplex * dummy;
	struct rohanNetwork * rnSrc, * rnDest ;
	//struct rohanLayer * rlSrc;
	//long LQTY, LLAST, LSIZE, 
	long SECSIZE;

	//printf("=> Copy Network %c =>\n", Direction);
	rnSrc=(rSes.rNet);
	long SIZE = sizeof(*rSes.rNet);
	//printf("%08lX %7ld %s\n", crc32buf( (char*)rnSrc, SIZE ), SIZE, "rNet");
		
	cudaGetSymbolAddress( (void**)&rnDest, "devNet" ); /*! get ptr into devspace for network structure */
		mCheckCudaWorked
	//LQTY = rnSrc->iLayerQty ; 
	//	LLAST = LQTY - 1 ;
	//LSIZE = sizeof(rohanLayer) * LQTY ;
	kind=cudaMemcpyHostToDevice;
		
	// mold dev memspace for sector table and copy contents
	SECSIZE = rnSrc->iSectorQty * sizeof(cuDoubleComplex);
	cudaMalloc( (void**)&rnSrc->gpuSectorBdry, SECSIZE );
		mCheckCudaWorked
		rSes.lMemStructAlloc = rSes.lMemStructAlloc || RNETbdry;
	cudaMemcpy( rnSrc->gpuSectorBdry, rnSrc->cdcSectorBdry, SECSIZE , kind );
		mCheckCudaWorked
	if(crc32buf( (char*)rnSrc->cdcSectorBdry, SECSIZE )!=knlCRC32Buf( (char*)rnSrc->gpuSectorBdry, SECSIZE ))
		printf("FAILURE copying sector table\n");
	//else
	//	printf("sector table faithfully copied\n");

	TransferNet(rSes, 'D');

	return 0;
}


long cDeviceTeam::TransferNet(struct rohanContext& rSes, char Direction)
{mIDfunc/*! copy rNet params to dev mem */
	long SIZE = sizeof(rohanNetwork);
	//printf("=> Transfer Net %c =>\n", Direction);
	if(Direction=='D'||Direction=='d') {
		cudaMemcpyToSymbol( "devNet", rSes.rNet, SIZE ); //, 0, kind );
			mCheckCudaWorked
	}
	else {
		// retrieve net params from device
		cudaMemcpyFromSymbol( rSes.rNet, "devNet", SIZE ); //, 0, kind );
			mCheckCudaWorked
	}
	if(crc32buf( (char*)rSes.rNet, SIZE )!=knlCRC32Buf( (char*)rSes.devNet, SIZE ))
			printf("FAILURE copying Net params\n");
	//else
	//	printf("%c net params faithfully copied\n", Direction);

	return 0;
}


long cDeviceTeam::CopyLearnSet(struct rohanContext& rSes, char Direction)
{mIDfunc/*! copies learning set structures, contents, andvmember values to device memory space */
	cudaMemcpyKind kind; //cuDoubleComplex * dummy;
	struct rohanLearningSet * rlSrc;
	long IQTY, OQTY, INSIZED, OUTSIZED, INSIZECX, OUTSIZECX;
	
	//setup dimension values
	IQTY = rSes.rLearn->iInputQty+1 ;
	INSIZED = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(double) ;
	INSIZECX = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(cuDoubleComplex) ;
	OQTY = rSes.rLearn->iOutputQty+1; 
	OUTSIZED = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(double);
	OUTSIZECX = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(cuDoubleComplex);
	
	//printf("=> Copy Learning Set %c =>\n", Direction);
	rlSrc=(rSes.rLearn);
	long SIZE = sizeof(*rSes.rLearn);
	//printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc, SIZE ), SIZE, "rLearn");
	
	if(Direction=='D'||Direction=='d') {
		kind=cudaMemcpyHostToDevice;
		// gpudXInputs
		cudaMalloc( (void**)&rlSrc->gpudXInputs, INSIZED ); /*! mold devspace for scalar input signals */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudXInputs, rlSrc->dXInputs, INSIZED, kind); /*! fill the devspace mold with hostspace values */
			mCheckCudaWorked
		// gpudDOutputs
		cudaMalloc( (void**)&rlSrc->gpudDOutputs, OUTSIZED ); /*! 2D desired scalar outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudDOutputs, rlSrc->dDOutputs, OUTSIZED, kind); 
			mCheckCudaWorked
		// gpudYEval
		cudaMalloc( (void**)&rlSrc->gpudYEval   , OUTSIZED ); /*! 2D yielded scalar outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudYEval   , rlSrc->dYEval   , OUTSIZED, kind); 
			mCheckCudaWorked
		// gpudAltYEval
		cudaMalloc( (void**)&rlSrc->gpudAltYEval, OUTSIZED ); /*! 2D yielded scalar outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudAltYEval, rlSrc->dAltYEval, OUTSIZED, kind); 
			mCheckCudaWorked
		// gpudSqrErr
		cudaMalloc( (void**)&rlSrc->gpudSqrErr, OUTSIZED ); /* array for intermediate RMSE totals, changed from 1024 to OUTSIZED on 1/8/12 */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudSqrErr, rlSrc->dSqrErr, OUTSIZED , kind );
			mCheckCudaWorked
		// gpuXInputs
		cudaMalloc( (void**)&rlSrc->gpuXInputs, INSIZECX ); /*! mold devspace for complex input signals */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpuXInputs, rlSrc->cdcXInputs, INSIZECX, kind); /*! fill the devspace mold with hostspace values */
			mCheckCudaWorked
		// gpuDOutputs
		cudaMalloc( (void**)&rlSrc->gpuDOutputs, OUTSIZECX ); /*! 2D desired complex outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpuDOutputs, rlSrc->cdcDOutputs, OUTSIZECX, kind); 
			mCheckCudaWorked
		// gpuYEval
		cudaMalloc( (void**)&rlSrc->gpuYEval   , OUTSIZECX ); /*! 2D yielded complex outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpuYEval   , rlSrc->cdcYEval   , OUTSIZECX, kind); 
			mCheckCudaWorked
		// gpuAltYEval
		cudaMalloc( (void**)&rlSrc->gpuAltYEval, OUTSIZECX ); /*! 2D yielded complex outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpuAltYEval, rlSrc->cdcAltYEval, OUTSIZECX, kind); 
			mCheckCudaWorked
		
		if(0){ // checksums to verify faithful transfer
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dXInputs, INSIZED ), INSIZED, "dXInputs");
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudXInputs, INSIZED ), INSIZED, "gpudXInputs");
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dDOutputs, OUTSIZED ), OUTSIZED, "dDOutputs");
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudDOutputs, OUTSIZED ), OUTSIZED, "gpudDOutputs");
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dYEval, OUTSIZED ), OUTSIZED, "dYEval");
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudYEval, OUTSIZED ), OUTSIZED, "gpudYEval");
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dAltYEval, OUTSIZED ), OUTSIZED, "dAltYEval");
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudAltYEval, OUTSIZED ), OUTSIZED, "gpudAltYEval");
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dSqrErr, OUTSIZED ), OUTSIZED, "dSqrErr"); // made OUTSIZED by JAW 1/8/12
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudSqrErr, OUTSIZED ), OUTSIZED, "gpudSE102");
		}
		// store array dimensions
		rSes.rLearn->IQTY=IQTY;
		rSes.rLearn->OQTY=OQTY;
		rSes.rLearn->INSIZED=INSIZED;
		rSes.rLearn->OUTSIZED=OUTSIZED;
		rSes.rLearn->INSIZECX=INSIZECX;
		rSes.rLearn->OUTSIZECX=OUTSIZECX;
		
		// publish Learn struct to device
		rSes.rLearn->lSampleIdxReq=77777;
		
		//printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc, SIZE ), SIZE, "rLearn");
		//printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rSes.devLearn, SIZE ), SIZE, "devLearn");
		
		//ShowMeLS(rSes, 0);
		
		cudaMemcpyToSymbol( "devLearn", rSes.rLearn, sizeof(*rSes.rLearn) ); //, 0, kind );
			mCheckCudaWorked
		//printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rSes.devLearn, SIZE ), SIZE, "devLearn");
	}
	else {
		kind=cudaMemcpyDeviceToHost;
	}
	if(crc32buf( (char*)rlSrc, SIZE )!=knlCRC32Buf( (char*)rSes.devLearn, SIZE ))
		printf("FAILURE copying samples\n");
	//else
	//	printf("samples faithfully copied\n");

	return 0;
}

long cDeviceTeam::LetUnHitch(struct rohanContext& rSes)
{mIDfunc/*! \callgraph \callergraph free device memory structures to Team */
	if(bHitched){ // check for hitched state
		if(bTaut) // check for taut/slack state
			LetSlack(rSes); // if taut, go slack before unhitching
		// BEGIN free nw data structures
		struct rohanNetwork * rnSrc=(rSes.rNet);

		cudaFree( rnSrc->gpuSectorBdry);
			mCheckCudaWorked
		printf("cdt: Net Arch clear\n");	

		//BEGIN freeing rdLearn structures
		struct rohanLearningSet * rlSrc = rSes.rLearn;
			cudaFree( rlSrc->gpudXInputs);
				mCheckCudaWorked
			cudaFree( rlSrc->gpudDOutputs);
				mCheckCudaWorked
			cudaFree( rlSrc->gpudYEval);
				mCheckCudaWorked
			cudaFree( rlSrc->gpudAltYEval);
				mCheckCudaWorked
			cudaFree( rlSrc->gpudSqrErr);
				mCheckCudaWorked
			cudaFree( rlSrc->gpuXInputs);
				mCheckCudaWorked
			cudaFree( rlSrc->gpuDOutputs);
				mCheckCudaWorked
			cudaFree( rlSrc->gpuYEval);
				mCheckCudaWorked
			cudaFree( rlSrc->gpuAltYEval);
				mCheckCudaWorked
		printf("cdt: LearnSet clear\n");

		/// end members and structures
		printf("cDeviceTeam unhitched.\n");
		bHitched=FALSE;
		return TRUE;
	}
	else{
		printf("cDeviceTeam is already unhitched.\n");
		return FALSE;
	}
}


long cDeviceTeam::LetTaut(struct rohanContext& rSes)
{mIDfunc/*! \callgraph \callergraph update dev mem from host for epoch */;
	if(bHitched && bTaut==FALSE){ // check taut state first
		TransferContext(rSes, 'D');
		//TransferLayers(rSes, 'D');
		TransferNet(rSes, 'D');
		TransferOutputs(rSes, 'D');
		if(rSes.bConsoleUsed)printf("T-->\n");
		bTaut=TRUE;
	}
	else{
		if(bTaut)
			printf("cDeviceTeam already taut!\n");
		if(bHitched==FALSE)
			printf("cDeviceTeam is not hitched!\n");
	}
	
	return 0;
}


long cDeviceTeam::TransferLayers(struct rohanContext& rSes, char Direction)
{mIDfunc/*! \callgraph \callergraph copy rNet layer data to dev mem */
	cudaMemcpyKind kind; //cuDoubleComplex * dummy;
	struct rohanNetwork * rnSrc, * rnDest ;
	struct rohanLayer * rlSrc;
	long LQTY, LLAST, LSIZE;

	rnSrc=(rSes.rNet);
	long SIZE = sizeof(*rSes.rNet);
	//printf("%08lX %7ld %s\n", crc32buf( (char*)rnSrc, SIZE ), SIZE, "rNet");
		
	cudaGetSymbolAddress( (void**)&rnDest, "devNet" ); /*! get ptr into devspace for network structure */
		mCheckCudaWorked
	LQTY = rnSrc->iLayerQty ; 
		LLAST = LQTY - 1 ;
	LSIZE = sizeof(rohanLayer) * LQTY ;

	if (Direction=='D' || Direction=='D'){
		kind=cudaMemcpyHostToDevice;
		for (long L=1; L<=LLAST; ++L){
			//printf("----> Copy Layer %d %c ---->\n", L, Direction);
			long DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE;
			//setup dimension values
			DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = weights
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
			rlSrc=&(rSes.rNet->rLayer[L]);
			//gpuWeights
			cudaMemcpy( rlSrc->gpuWeights, rlSrc->Weights, WSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuXInputs
			cudaMemcpy( rlSrc->gpuXInputs, rlSrc->XInputs, DSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuZOutputs
			cudaMemcpy( rlSrc->gpuZOutputs, rlSrc->ZOutputs, NSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuDeltas 
			cudaMemcpy( rlSrc->gpuDeltas, rlSrc->Deltas, NSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//printf("-> layer %d transfered?\n", L);
		}
	}
	else{
		kind=cudaMemcpyDeviceToHost;
		for (long L=1; L<=LLAST; ++L){
			//printf("----> Copy Layer %d %c ---->\n", L, Direction);
			long DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE;
			//setup dimension values
			DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = weights
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
			rlSrc=&(rSes.rNet->rLayer[L]);
			//gpuWeights
			cudaMemcpy( rlSrc->Weights, rlSrc->gpuWeights, WSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuXInputs
			cudaMemcpy( rlSrc->XInputs, rlSrc->gpuXInputs, DSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuZOutputs
			cudaMemcpy( rlSrc->ZOutputs, rlSrc->gpuZOutputs, NSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuDeltas 
			cudaMemcpy( rlSrc->Deltas, rlSrc->gpuDeltas, NSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//printf("-> layer %d transfered?\n", L);
		}
	}
	//printf("cdt: Layers (weights only) copied %c\n", Direction);
	
	return 0;
}


long cDeviceTeam::LetSlack(struct rohanContext& rSes)
{mIDfunc/*! \callgraph \callergraph update dev mem from host for epoch */;
	if(bHitched && bTaut){ // check taut state first
		//TransferContext(rSes, 'H');
		//TransferLayers(rSes, 'H');
		TransferNet(rSes, 'H');
		TransferOutputs(rSes, 'H');
		if(rSes.bConsoleUsed)printf("<--S\n");
		bTaut=FALSE;
	}
	else{
		if(bTaut==FALSE)
			printf("cDeviceTeam already slack!\n");
		if(bHitched==FALSE)
			printf("cDeviceTeam is not hitched!\n");
	}
	return 0;
}


long cDeviceTeam::TransferOutputs(struct rohanContext& rSes, char Direction)
{mIDfunc/*! transfers contents of yielded output data strctures between memory spaces, usually dev to host */
	cudaMemcpyKind kind; //cuDoubleComplex * dummy;
	struct rohanLearningSet * rlSrc;
	long IQTY, OQTY, INSIZED, OUTSIZED, INSIZECX, OUTSIZECX;
	
	//setup dimension values
	IQTY = rSes.rLearn->iInputQty+1 ;
	INSIZED   = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(double) ;
	INSIZECX  = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(cuDoubleComplex) ;
	OQTY = rSes.rLearn->iOutputQty+1; 
	OUTSIZED  = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(double);
	OUTSIZECX = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(cuDoubleComplex);
	
	rlSrc=(rSes.rLearn);

	if(Direction=='D'||Direction=='d') {
		kind=cudaMemcpyHostToDevice; // CPU plain evals go to GPU alt evals, and vice versa
		// gpudYEval
		cudaMemcpy( rlSrc->gpudYEval   , rlSrc->dAltYEval   , OUTSIZED, kind); 
			mCheckCudaWorked
		// gpudAltYEval 
		cudaMemcpy( rlSrc->gpudAltYEval, rlSrc->dYEval      , OUTSIZED, kind); 
			mCheckCudaWorked
		// gpuYEval 
		cudaMemcpy( rlSrc->gpuYEval   , rlSrc->cdcAltYEval, OUTSIZECX, kind); 
			mCheckCudaWorked
		// gpuAltYEval 
		cudaMemcpy( rlSrc->gpuAltYEval, rlSrc->cdcYEval   , OUTSIZECX, kind); 
			mCheckCudaWorked
	}
	else {
		kind=cudaMemcpyDeviceToHost; // GPU plain evals go to CPU alt evals, and vice versa
		cudaMemcpy( rlSrc->dAltYEval   , rlSrc->gpudYEval   , OUTSIZED, kind); 
			mCheckCudaWorked
		cudaMemcpy( rlSrc->dYEval      , rlSrc->gpudAltYEval, OUTSIZED, kind); 
			mCheckCudaWorked
		cudaMemcpy( rlSrc->cdcAltYEval, rlSrc->gpuYEval   , OUTSIZECX, kind); 
			mCheckCudaWorked
		cudaMemcpy( rlSrc->cdcYEval   , rlSrc->gpuAltYEval, OUTSIZECX, kind); 
			mCheckCudaWorked
	}

	return 0;
}

long cDeviceTeam::GetEvalSingleSample( struct rohanContext& rSes, long lSampleIdxReq, char chMethod)
{mIDfunc/*! calculates NN outputs for a given sample with GPU method */
	if(chMethod=='c')
		return cuEvalSingleSampleBeta(rSes, lSampleIdxReq, *rSes.rNet, 0, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval);
	else // d for GPU device XX
		return 0;////return devEvalSingleSample(rSes, lSampleIdxReq);
}


double cDeviceTeam::CUDAverify(struct rohanContext& rSes)
{mIDfunc/// Checks for prsence of CUDA-enabled hardware
	double compCap=1.0, check; rSes.dMasterCalcVer=0.0;
	
	cudaGetDeviceCount(&(rSes.deviceCount) ); 
	if(rSes.deviceCount){
		for (int device = 0; device < rSes.deviceCount; ++device) { 
			//CUDAShowProperties(rSes, device, rSes.debugHandle);
			cudaGetDeviceProperties(&rSes.deviceProp, device); 
			check=rSes.deviceProp.major + rSes.deviceProp.minor * 0.1;
			if (check>compCap){
				compCap=check;
				if(check>rSes.dMasterCalcVer){
					rSes.dMasterCalcVer=check;
					rSes.iMasterCalcHw=device;
				}
			}
		}
		return compCap;
	}
	else 
		return 0; // not available
}


// following section borrowed from CUDA SDK, (C) Nvidia
// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
	CUresult error_result = cuDeviceGetAttribute( attribute, device_attribute, device );
    if (error_result != CUDA_SUCCESS) {
        shrLog( "cuDeviceGetAttribute returned %d\n-> %s\n", (int)error_result, getCudaDrvErrorString(error_result) );
		exit(0);
    }
}
// end borrowed section


void cDeviceTeam::CUDAShowProperties(struct rohanContext& rSes, int device, FILE* fShow)
{
	if(fShow!=NULL){
		// following section borrowed from CUDA SDK, (C) Nvidia
		fprintf(fShow,"\nDevice %d has compute capability %d.%d.\n", device, rSes.deviceProp.major, rSes.deviceProp.minor);
		fprintf(fShow,"  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", 
			(float)rSes.deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) rSes.deviceProp.totalGlobalMem);
		fprintf(fShow,"  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n", 
			rSes.deviceProp.multiProcessorCount, ConvertSMVer2Cores(rSes.deviceProp.major, rSes.deviceProp.minor),
			ConvertSMVer2Cores(rSes.deviceProp.major, rSes.deviceProp.minor) * rSes.deviceProp.multiProcessorCount);
		//int L2CacheSize;
		//getCudaAttribute<int>( &L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device );
		//if (L2CacheSize) {fprintf(fShow,"  L2 Cache Size:                                 %d bytes\n", L2CacheSize);}
		fprintf(fShow,"  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
			rSes.deviceProp.maxTexture1D, rSes.deviceProp.maxTexture2D[0], rSes.deviceProp.maxTexture2D[1],
			rSes.deviceProp.maxTexture3D[0], rSes.deviceProp.maxTexture3D[1], rSes.deviceProp.maxTexture3D[2]);
		fprintf(fShow,"  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
			rSes.deviceProp.maxTexture1DLayered[0], rSes.deviceProp.maxTexture1DLayered[1],
			rSes.deviceProp.maxTexture2DLayered[0], rSes.deviceProp.maxTexture2DLayered[1], rSes.deviceProp.maxTexture2DLayered[2]);
		fprintf(fShow,"  Total amount of constant memory:               %u bytes\n", rSes.deviceProp.totalConstMem); 
		fprintf(fShow,"  Total amount of shared memory per block:       %u bytes\n", rSes.deviceProp.sharedMemPerBlock);
		fprintf(fShow,"  Total number of registers available per block: %d\n", rSes.deviceProp.regsPerBlock);
		fprintf(fShow,"  Warp size:                                     %d\n", rSes.deviceProp.warpSize);
		fprintf(fShow,"  Maximum number of threads per block:           %d\n", rSes.deviceProp.maxThreadsPerBlock);
		fprintf(fShow,"  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
			rSes.deviceProp.maxThreadsDim[0],
			rSes.deviceProp.maxThreadsDim[1],
			rSes.deviceProp.maxThreadsDim[2]);
		fprintf(fShow,"  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
			rSes.deviceProp.maxGridSize[0],
			rSes.deviceProp.maxGridSize[1],
			rSes.deviceProp.maxGridSize[2]);
		// end borrowed section
		fprintf(fShow,"  sizeof devSes, devLearn, devNet:               %d, %d, %d\n", sizeof(rohanContext), sizeof(rohanLearningSet), sizeof(rohanNetwork) );
	}
}