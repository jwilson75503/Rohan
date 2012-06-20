/* Includes, cuda */
#include "stdafx.h"
#include <boost/timer/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp> //include all types plus i/o
#include <boost/date_time/gregorian/gregorian.hpp> //include all types plus i/o
#include <boost/program_options.hpp>
using namespace boost::timer;
using namespace boost::posix_time;
using namespace boost::gregorian;
namespace po = boost::program_options;
//#include <boost/filesystem.hpp>
//using namespace boost::filesystem;
#if defined (_INTEGRAL_MAX_BITS) &&  _INTEGRAL_MAX_BITS >= 64
typedef signed __int64 int64;
typedef unsigned __int64 uint64;
#else
#error __int64 type not supported
#endif

extern int gDebugLvl, gDevDebug, gTrace;
extern long bCUDAavailable;
extern float gElapsedTime, gKernelTimeTally;

//////////////// class cDrover begins ////////////////

void cDrover::ShowMe()
{
	//ShowMeSes(* rSes, false);
	printf("Am Volga boatman.\n");
}


long cDrover::SetContext( rohanContext& rC, rohanLearningSet& rL, rohanNetwork& rN)
{/// enables pointer access to master context struct, sets up pointers to toher objects of interest
	rSes = &rC;
	// pointers to learning set and network object copies in host memory space are recorded
	rC.rLearn = &rL;
	rC.rNet = &rN; 
	// pointers to learning set and network object copies in dev memory space are recorded
	cudaGetSymbolAddress( (void**)&rC.devSes, "devSes" );
		mCheckCudaWorked
	cudaGetSymbolAddress( (void**)&rC.devLearn, "devLearn" );
		mCheckCudaWorked
	cudaGetSymbolAddress( (void**)&rC.devNet, "devNet" );
		mCheckCudaWorked

	return 0;
}


long cDrover::SetDroverBargeAndTeam( class cBarge& cbB, class cDeviceTeam& cdtT)
{mIDfunc /// sets pointers to hitch barge to team and mount driver on barge
	Barge = &cbB;
	Team = &cdtT;
	Barge->SetDrover(this);
	Barge->SetTeam(Team);
	Team->SetDrover(this);
	Team->SetBarge(Barge);
	return 0;
}


long cDrover::DoAnteLoop(int argc, char * argv[])
{mIDfunc /// This function prepares all parameters and data structures necesary for learning and evaluation.
	int iReturn=1;
	
	SetProgOptions( *rSes, argc, argv);
	ObtainGlobalSettings( *rSes);		
	printf("Rohan v%s Neural Network Simulator\n", VERSION);
	if(iReturn*=ShowDiagnostics( *rSes, *rSes->rNet ))
			iReturn*=Barge->ShowDiagnostics();

	return iReturn;
}


int cDrover::SetProgOptions(struct rohanContext& rSes, int argc, char * argv[])
{mIDfunc /// Declare the supported options.
    try {
		// Declare a group of options that will be 
		// allowed only on command line
		po::options_description generic("Generic options");
		generic.add_options()
			("version,v", "print version string")
			("help,h", "produce help message")
			("learn,l", po::value< vector<string> >(), "pursue backprop training in pursuit of target RMSE given MAX criterion")
			("eval,e", po::value< vector<string> >(), "evaluate samples and report")
			("tag,t", po::value< vector<string> >(), "tag session with an identifying string")
            ;
		    
		// Declare a group of options that will be 
		// allowed both on command line and in
		// config file
		po::options_description config("Configuration");
		config.add_options()
            ("net,n", po::value< vector<string> >(), "network sectors, inputs, 1st hidden layer, 2nd hidden layer, outputs")
			("samples,s", po::value< vector<string> >(), "text file containing sample input-output sets")
			("weights,w", po::value< vector<string> >(), ".wgt file containing complex weight values")
			("include-path,I", 
				 po::value< vector<string> >()->composing(), 
				 "include path")
			;

		// Hidden options, will be allowed both on command line and
		// in config file, but will not be shown to the user.
		po::options_description hidden("Hidden options");
		hidden.add_options()
			("input-file", po::value< vector<string> >(), "input file")
			; 

		po::options_description cmdline_options;
		cmdline_options.add(generic).add(config).add(hidden);

		po::options_description config_file_options;
		config_file_options.add(config).add(hidden);

		po::options_description visible("Allowed options");
		visible.add(generic).add(config);

        po::variables_map vm;        
        po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << visible << "\n";
            return 1;
        }

        if (vm.count("version")) {
            cout << VERSION << ".\n";
        }
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

    return 0;
}


long cDrover::ObtainGlobalSettings(struct rohanContext& rSes)
{mIDfunc /// sets initial and default value for globals and settings
	int iReturn=1;
	rSes.bConsoleUsed=true;
	//	globals
	gTrace=0; 
	gDebugLvl=0; 
	// session accrual
	rSes.iWarnings=0; rSes.iErrors=0; 
	rSes.lSampleQtyReq=0;

	char sPath[MAX_PATH];
	GetUserDocPath(sPath);
	
	sprintf(rSes.sRohanVerPath, "%s\\Rohan_%s", sPath, VERSION);
		//cout << rSes.sRohanVerPath << "\n";

	if(DirectoryEnsure(rSes.sRohanVerPath)){
		sprintf(sPath, "%s\\RohanLog.txt", rSes.sRohanVerPath);
		rSes.ofsRLog=new ofstream(sPath, std::ios::app|std::ios::out); 
		*(rSes.ofsRLog) << "\tSTART Rohan v" << VERSION << " Neural Network Application\n";
		using namespace boost::posix_time; 
		ptime now = second_clock::local_time(); //use the clock

		AsciiFileHandleWrite(rSes.sRohanVerPath, "DevBucket.txt", &(rSes.deviceBucket));
		//fprintf(rSes.deviceBucket, "%s\tSTART Rohan v%s Neural Network Application\n", "to_simple_string(now)", VERSION);
		AsciiFileHandleWrite(rSes.sRohanVerPath, "HostBucket.txt", &(rSes.hostBucket));
		//fprintf(rSes.hostBucket, "%s\tSTART Rohan v%s Neural Network Application\n", "to_simple_string(now)", VERSION);
	}
	else {
		errPrintf("Directory %s could not be createed\n", rSes.sRohanVerPath);
		++rSes.iWarnings;
	}
	// memory structures
	rSes.lMemStructAlloc=0;
	// session modes
	rSes.bRInJMode=false; 
	rSes.bRMSEon=true; 
	rSes.iEpochLength=1000; 
	rSes.iEvalBlocks=128; 
	rSes.iEvalThreads=128; 
	rSes.iBpropBlocks=1; 
	rSes.iBpropThreads=32; 
	rSes.iSaveInputs=0 /* include inputs when saving evaluations */;
	rSes.iSaveOutputs=0 /* include desired outputs when saving evaluations */;
	rSes.iSaveSampleIndex=0 /* includes sample serials when saving evaluations */;
	strcpy(rSes.sSesName,"DefaultSession");
	
	// learning set modes
	rSes.rLearn->bContInputs=true; 
	rSes.rLearn->iContOutputs=false; 

	// network modes
	rSes.rNet->bContActivation=true; 
	rSes.dHostRMSE=0.0;
	rSes.dDevRMSE=0.0;

		if(gTrace) cout << "Tracing is ON.\n" ;
		if (gDebugLvl){
			cout << "Debug level is " << gDebugLvl << "\n" ;
			cout << "Session warning and session error counts reset.\n";
			rSes.rNet->bContActivation ? cout << "Activation default is CONTINUOUS.\n" : cout << "Activation default is DISCRETE.\n"; 
			// XX defaulting to false makes all kinds of heck on the GPU
			rSes.bRInJMode ? cout << "Reversed Input Order is ON.\n" : cout << "Reversed Input Order is OFF.\n"; 
			// this is working backward for some reason 2/08/11 // still fubared 3/7/12 XX
			cout << "RMSE stop condition is ON. XX\n"; //
			cout << "Epoch length is " << rSes.iEpochLength << " iterations.\n";
			cout << rSes.iEvalBlocks << " EVAL Blocks per Kernel, " << rSes.iEvalThreads << " EVAL Threads per Block.\n";
			cout << rSes.iBpropBlocks << " BPROP Blocks per Kernel, " << rSes.iBpropThreads << " BPROP Threads per Block.\n";
			cout << "Continuous Inputs true by DEFAULT.\n";
			cout << "Continuous Outputs true by DEFAULT.\n";
		}

	//strcpy(rSes->sCLargsOpt, "blank");
	//strcpy(rSes->sConfigFileOpt,"blank");
	//strcpy(rSes->sConsoleOpt,"blank");
	//strcpy(rSes->sLearnSetSpecOpt,"blank");

	if (Team->CUDAverify(rSes)>=2.0){
		cutilSafeCall( cudaSetDevice(rSes.iMasterCalcHw) ); /// all cuda calls to run on first device of highest compute capability device located
		if (gDebugLvl) cout << "CUDA present, device " << rSes.iMasterCalcHw << " selected." << endl;
	}
	else {
		if (rSes.dMasterCalcVer>1.0)
			fprintf(stderr, "Warning: CUDA hardware below Compute Capability 2.0.\n");
		else
			fprintf(stderr, "Warning: No CUDA hardware or no CUDA functions present.\n");
		rSes.iMasterCalcHw=-1;
		++rSes.iWarnings;
		iReturn=0;
	}

	return iReturn;
}


long cDrover::ShowDiagnostics(struct rohanContext& rSes, struct rohanNetwork& rNet)
{mIDfunc /// show some statistics, dump weights, and display warning and error counts
	double devdRMSE=0.0, hostdRMSE=0.0;
	int iReturn=1;
	cuDoubleComplex keepsakeWts[MAXWEIGHTS]; // 16 x 2048
	
	if(rSes.bConsoleUsed)AskSampleSetName(rSes);
	if(iReturn=Barge->ObtainSampleSet(rSes)){
		iReturn=Barge->DoPrepareNetwork(rSes);
		cudaMemcpy( keepsakeWts, rSes.rNet->Wt, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // backup weights for post-test restoration
		printf("(backup wt %08lX)\n", crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS ) ); // weight check
	
		Team->LetHitch(rSes);
		Team->LetSlack(rSes);
	
		RmseEvaluateTest(rSes, rNet, 2 , 0);
		// some illustrative default values
		rSes.dTargetRMSE=floor((sqrt(rSes.dHostRMSE/10)*10)-1.0)+1.0;
		rSes.dMAX=rSes.dTargetRMSE-2;
		// more tests
		//ClassifyTest(rSes, rNet, 8, 0 );
		//rSes.dTargetRMSE=0.0;
		//rSes.dMAX=0.0;
		//	SetDevDebug(1); gDevDebug=1;
		BackPropTest(rSes, rNet, 4, 512, 0);
		//	SetDevDebug(0);  gDevDebug=0;
		//cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//BackPropTest(rSes, rNet, 1024, 64, 0);
		//cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//BackPropTest(rSes, rNet, 10, 128, 0);
		//cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//BackPropTest(rSes, rNet, 32, 256, 0);
		//cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//BackPropTest(rSes, rNet, 10, 512, 0);
		//RmseEvaluateTest(rSes, rNet, 1 , 1);
		
		cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		printf("(restrd wt %08lX)\n", crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS ) ); // weight check
	}
	
	if (rSes.iWarnings) fprintf(stderr, "Diagnosis: %d warnings.\n", rSes.iWarnings);
	if (rSes.iErrors) fprintf(stderr, "Diagnosis: %d operational errors.\n", rSes.iErrors);

	return iReturn;
}

double cDrover::RmseEvaluateTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iSampleQty)
{mIDfunc /// runs tests for RMSE and evaluation on both host and GPU
	int iDifferent=0; double dDifferent=0.0; float fDevTime=0.0;
	boost::timer::cpu_timer tHost, tDev;
	boost::timer::cpu_times elapHost, elapDev;
		// perform a warm-up host eval to eliminate the always-longer first one, return true number of samples, prepare timer to resume @ 0.0;
		printf("WARMUP:\n");
			iSampleQty=cuEvalNNLearnSet(rSes, iSampleQty);
			RmseNN(rSes, iSampleQty); // update dHostRMSE
		tHost.start();
		tHost.stop();
			Team->LetTaut(rSes);
			Team->GetRmseNN(rSes, iSampleQty);
			Team->LetSlack(rSes);
		tDev.start();
		tDev.stop();
		char sLog0[80]; sprintf(sLog0, "BEGIN RMSE/EVALUATE TEST: %d TRIALS, %d SAMPLES", iTrials, iSampleQty);
		printf("\n\n%s\n\n", sLog0); RLog( rSes, sLog0);
		printf("-------------------------------\n");
	
	for(int i=1; i<=iTrials; ++i){
		//reset values
		rSes.dDevRMSE = rSes.dHostRMSE = 0.0;
		// begin dev eval test
		Team->LetTaut(rSes);
		tDev.resume();
			//printf(">>DEVICE: RMSE = %f\n", Team->GetRmseNN(rSes, iSampleQty));
			Team->GetRmseNN(rSes, iSampleQty);
		fDevTime+=gElapsedTime; // device times are roughly equal to serial overhead; kernel launchers record time in global variable for later pickup
		tDev.stop();
		Team->LetSlack(rSes);
		// end dev eval test

		//begin host eval test
		//printf("HST:");
		{
			boost::timer::auto_cpu_timer o;
			tHost.resume();
			cuEvalNNLearnSet(rSes, iSampleQty);
			RmseNN(rSes, iSampleQty); // update dHostRMSE
			tHost.stop();
		}
		// end host eval test

		iDifferent += OutputValidate(rSes);
		printf("BOTH: %d differences found on verify.\n", iDifferent);
		dDifferent += rSes.dDevRMSE - rSes.dHostRMSE;
		printf("BOTH: delta RMSE %f += %f - %f\n", dDifferent, rSes.dDevRMSE, rSes.dHostRMSE);
		printf("-------------------------------%d\n", i);
	}
	elapHost=tHost.elapsed();
	elapDev =tDev.elapsed();
	int64 denominator = iTrials*100000; // convert to tenths of milliseconds
	int64 quotientHost = elapHost.wall / denominator;
	int64 quotientDev  = elapDev.wall  / denominator;
	double dAvgTimeHost = (double)quotientHost; 
	double dAvgTimeDev = (double)quotientDev; 
	char sLog1[80]; sprintf(sLog1, "Host/Serial mean performance over %d runs: %.1f ms", iTrials, dAvgTimeHost/10);
	char sLog2[80]; sprintf(sLog2, "Dev/CUDA    mean performance over %d runs: %.1f ms", iTrials, fDevTime/iTrials);
	char sLog3[14]; sprintf(sLog3, ( (iDifferent || abs(dDifferent)>.001 ) ? "EVALUATE FAIL" : "EVALUATE PASS" ) );
	printf(" %s\n %s\n\n%s\n\n", sLog1, sLog2, sLog3);
	//sprintf(sLog1, "%s %s", sLog3, sLog1); 
	RLog(rSes, sLog1);
	//sprintf(sLog2, "%s %s", sLog3, sLog2); 
	RLog(rSes, sLog2);
	RLog(rSes, sLog3);
	return iDifferent+dDifferent;
}

int cDrover::ClassifyTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iSampleQty)
{mIDfunc /// runs classification tests on both host and GPU
	int iDeviceTrainable, iHostTrainable, iMargin=0;  float fDevTime=0.0;
	boost::timer::cpu_timer tHost, tDev;
	boost::timer::cpu_times elapHost, elapDev;
		// perform a warm-up host eval to eliminate the always-longer first one, return true number of samples, prepare timer to resume @ 0.0;
		printf("WARMUP:\n");
			iSampleQty=cuEvalNNLearnSet(rSes, iSampleQty);
			RmseNN(rSes, iSampleQty); // update dHostRMSE
		tHost.start();
		tHost.stop();
			Team->LetTaut(rSes);
			Team->GetRmseNN(rSes, iSampleQty);// evaluation is now included in classification tests
			Team->LetSlack(rSes);
		tDev.start();
		tDev.stop();
	char sLog0[80]; sprintf(sLog0, "BEGIN CLASSIFY TEST: %d TRIALS, %d SAMPLES", iTrials, iSampleQty);
	printf("\n\n%s\n\n", sLog0); RLog( rSes, sLog0);
	printf("-------------------------------\n");
	for(int i=1; i<=iTrials; ++i){
		// begin trainable sample test DEVICE
		Team->LetTaut(rSes);
		tDev.resume();
			gKernelTimeTally=0.0; //reset global kernel time tally
			// evaluation is now integrated in device classification tests
			iMargin+=iDeviceTrainable=Team->LetTrainNNThresh(rSes, iSampleQty, 1, 'E', rSes.dTargetRMSE, rSes.iEpochLength);
			fDevTime+=gKernelTimeTally; // device times are roughly equal to serial overhead; kernel launchers record time in global variable for later pickup
		tDev.stop();
		Team->LetSlack(rSes);
		printf("HST:");
		{
			boost::timer::auto_cpu_timer o;
			tHost.resume();
				cuEvalNNLearnSet(rSes, iSampleQty); // evaluation is now included separately in host classification tests
				iMargin -= iHostTrainable=TrainNNThresh(rSes, false, iSampleQty);
			tHost.stop();
		}
		printf("BOTH: delta trainable %d += %d - %d\n", iMargin, iDeviceTrainable, iHostTrainable);
		printf("-------------------------------%d\n", i);
		iDeviceTrainable=iHostTrainable=0;
	}

	elapHost=tHost.elapsed();
	elapDev =tDev.elapsed();
	int64 denominator = iTrials*100000; // convert to tenths of milliseconds
	int64 quotientHost = elapHost.wall / denominator;
	int64 quotientDev  = elapDev.wall  / denominator;
	double dAvgTimeHost = (double)quotientHost; 
	double dAvgTimeDev = (double)quotientDev; 
	
	char sLog1[80]; sprintf(sLog1, "Host/Serial mean performance over %d runs: %.1f ms", iTrials, dAvgTimeHost/10);
	char sLog2[80]; sprintf(sLog2, "Dev/CUDA    mean performance over %d runs: %.1f ms", iTrials, fDevTime/iTrials);
	char sLog3[14]; sprintf(sLog3, ( iMargin ? "CLASSIFY FAIL" : "CLASSIFY PASS" ) );
	printf(" %s\n %s\n\n%s\n\n", sLog1, sLog2, sLog3);
	//sprintf(sLog1, "%s %s", sLog3, sLog1); 
	RLog(rSes, sLog1);
	//sprintf(sLog2, "%s %s", sLog3, sLog2); 
	RLog(rSes, sLog2);
	RLog(rSes, sLog3);
	return (iMargin);
}


double cDrover::BackPropTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iThreads, int iSampleQty)
{mIDfunc /// runs tests for backward propagation on both host and GPU
	double dDifferent=0.0;
	int iDeviceTrainable, iHostTrainable, iMargin=0, oldThreads=rSes.iBpropThreads;  float fDevTime=0.0;
	boost::timer::cpu_timer tHost;//, tDev;
	boost::timer::cpu_times elapHost;//, elapDev;
	rSes.iBpropThreads=iThreads;
		// perform a warm-up host eval to eliminate the always-longer first one, return true number of samples, prepare timer to resume @ 0.0;
		printf("WARMUP:\n");
			iSampleQty=cuEvalNNLearnSet(rSes, iSampleQty);
			RmseNN(rSes, iSampleQty); // update hostRMSE
		tHost.start();
		tHost.stop();
			Team->LetTaut(rSes);
			Team->GetRmseNN(rSes, iSampleQty);// evaluation is now included in classification tests
			Team->LetSlack(rSes);

	char sLog0[80]; sprintf(sLog0, "BEGIN BACKPROP TEST: %d TRIALS, %d THREADS %d SAMPLES", iTrials, iThreads, iSampleQty);
	printf("\n\n%s\n\n", sLog0); RLog( rSes, sLog0);
	printf("-------------------------------\n");
	Team->LetTaut(rSes);
	for(int i=1; i<=iTrials; ++i){
		// begin BACKPROPagation test DEVICE
			gKernelTimeTally=0.0; //reset global kernel time tally
			// evaluation is now integrated in device classification tests
			iMargin+=iDeviceTrainable=Team->LetTrainNNThresh( rSes, iSampleQty, 1, 'R', rSes.dTargetRMSE, 1); // backprop all samples, output #1, revise wts, target RMSE, epoch=single iteration YY change E to R
			dDifferent += Team->GetRmseNN(rSes, iSampleQty);
			fDevTime+=gKernelTimeTally; // device times are roughly equal to serial overhead; kernel launchers record time in global variable for later pickup
		//printf(">>DEVICE: %d samples\n", lSamplesBpropDev);
		//printf(">>DEVICE: RMSE=%f\n", rSes.dDevRMSE);
		// end device test

		// begin BACKPROPagation test HOST
		//conPrintf("HST:");
		{	
			boost::timer::auto_cpu_timer o;
			tHost.resume();
				cuEvalNNLearnSet(rSes, iSampleQty); // evaluation is now included separately in host classification tests
				iMargin -= iHostTrainable=TrainNNThresh( rSes, true, iSampleQty); // YY change false to true
				cuEvalNNLearnSet( rSes, iSampleQty); // re-revaluate learnset
				dDifferent -= RmseNN( rSes, iSampleQty); // update RMSE
			tHost.stop();
		}
		// end host test
		printf("BOTH: delta RMSE %f += %f - %f", dDifferent, rSes.dDevRMSE, rSes.dHostRMSE);
		//printf("BOTH: delta trainable %d += %d - %d\n", iMargin, iDeviceTrainable, iHostTrainable);
		printf("----------------------%d\n", i);
		iDeviceTrainable=iHostTrainable=0;
	}
	Team->LetSlack(rSes);
	elapHost=tHost.elapsed();
	int64 denominator = iTrials*100000; // convert to tenths of milliseconds
	int64 quotientHost = elapHost.wall / denominator;
	double dAvgTimeHost = (double)quotientHost; 
	rSes.iBpropThreads=oldThreads;
	char sLog1[80]; sprintf(sLog1, "Host/Serial mean performance over %d runs: %.1f ms", iTrials, dAvgTimeHost/10); //converted from tenths of ms to full ms
	char sLog2[80]; sprintf(sLog2, "Dev/CUDA    mean performance over %d runs: %.1f ms", iTrials, fDevTime/iTrials);
	char sLog3[14]; sprintf(sLog3, ( (iMargin || abs(dDifferent)>.001 ) ? "BACKPROP FAIL" : "BACKPROP PASS" ) );
	printf(" %s\n %s\n\n%s\n\n", sLog1, sLog2, sLog3);
	RLog(rSes, sLog1);
	RLog(rSes, sLog2);
	RLog(rSes, sLog3);
	return (iMargin+dDifferent);
}

long cDrover::AskSampleSetName(struct rohanContext& rSes)
{mIDfunc /// chooses the learning set to be worked with Ante-Loop
	int iReturn=0; 
	//rSes.rLearn->bContInputs=false;
	//rSes.rLearn->iContOutputs=(int)false;
	//cout << "Samples treated as discrete or continuous by fractionality. XX" << endl;

	printf("Enter 0 for 10K-set, 9-36-1 weights\n\t 1 for 3-set, 331 weights\n\t 2 for 150-set, no wgts\n\t 3 for 4-set, 2x21 wgts\n\t 4 for 2-1 rand weights");
	printf("\n\t 5 for 416 samples x 200 inputs\n\t 6 for 10k-set, 9-45-1 weights\n\t 7 for 10k-set, 9-54-1 weights\n");
	std::cin >> gDebugLvl;
	switch ( gDebugLvl % 10) {
		case 0:
		  rSes.rLearn->sLearnSet="AirplanePsDN1W3S10k.txt";
		  break;
		case 1:
		  rSes.rLearn->sLearnSet="trivial3331.txt";
		  break;
		case 2:
		  rSes.rLearn->sLearnSet="iris.txt";
		  break;
		case 3:
		  rSes.rLearn->sLearnSet="trivial221.txt";
		  break;
		case 4:
		  rSes.rLearn->sLearnSet="trivial3.txt";	
		  break;
		case 5:
		  rSes.rLearn->sLearnSet="PC-63-32-200-LearnSet.txt";
		  break;
		case 6:
		  rSes.rLearn->sLearnSet="LenaPsDN2W3S10k.txt";
		  break;
		case 7:
		  rSes.rLearn->sLearnSet="LenaPsUN2W3S10k.txt";
		  break;
		default:
		  rSes.rLearn->sLearnSet="iris.txt";
		  break;
	}
	rSes.gDebugLvl=gDebugLvl/=10; // drop final digit
	if (gDebugLvl) fprintf(stderr, "Debug level is %d.\n", gDebugLvl);
	return iReturn;
}


long cDrover::DoMainLoop(struct rohanContext& rSes)
{mIDfunc /// Trains a weight set to more closely reproduce the sampled outputs from the sampled inputs, and other options.
	int iReturn=0, iSelect=1;
	cout << "Main duty loop begin." << endl;
	
	while(iSelect){
		//Team->LetSlack(rSes);
		//Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
		//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
		iSelect=DisplayMenu(0, rSes);
		if (iSelect==1) iReturn=BeginSession(rSes); // new or resume session
		if (iSelect==2) iReturn=GetNNTop(rSes);
		//if (iSelect==3) iReturn=ReGetSampleSet(rSes); XX
		if (iSelect==4) iReturn=GetWeightSet(rSes);
		if (iSelect==5) iReturn=this->LetInteractiveEvaluation(rSes);
		if (iSelect==6) iReturn=this->LetInteractiveLearning(rSes);
		if (iSelect==7) iReturn=cuPreSaveNNWeights(rSes);
		if (iSelect==8) {iReturn=cuRandomizeWeightsBlock(rSes); 
						Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H'); // this is performed on the host
						//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
						RmseNN(rSes, rSes.lSampleQtyReq);
		}
		if (iSelect==9) iReturn=this->LetUtilities(rSes);
		
		//SilentDiagnostics(rSes);
	}
	return 0;
}


int cDrover::DisplayMenu(int iMenuNum, struct rohanContext& rSes)
{mIDfunc
	char a='.';
	//refresh RMSE
	cuEvalNNLearnSet(rSes, rSes.lSampleQtyReq);
	RmseNN(rSes, rSes.lSampleQtyReq);
	
	//list menu items
	if(iMenuNum==0){
		printf("\n1 - Label session \"%s\"", rSes.sSesName);
		printf("\n2 X Network topology setup");
		printf("\n3 / Sample set load");
		printf("\n4 - Weight set load");
		printf("\n5 - Evaluation Feed Forward");
		printf("\n6 - Learning");
		printf("\n7 - Save weights");
		printf("\n8 - Randomize weights");
		printf("\n9 - Utilities");
		printf("\n0 - Quit");
		CLIbase(rSes);
	}
	if(iMenuNum==50){
		printf("\n1 - Include inputs: %d", rSes.iSaveInputs);
		printf("\n2 - Include outputs: %d", rSes.iSaveOutputs);
		printf("\n3 - Include sample index: %d", rSes.iSaveSampleIndex);
		printf("\n4 - change Samples used: %d", rSes.lSampleQtyReq);
		printf("\n5 - host serial evaluation");
		printf("\n6 - Asynchronous device commence");
		printf("\n7 - change Blocks per kernel: %d", rSes.iEvalBlocks);
		printf("\n8 - change Threads per block: %d", rSes.iEvalThreads);
		printf("\n9 - Save evals");
		printf("\n0 - Quit");
		CLIbase(rSes);
	}
	if(iMenuNum==60){
		printf("\n1 - change Target RMSE: % #3.3g", rSes.dTargetRMSE);
		printf("\n2 - change MAX error: % #3.3g", rSes.dMAX);
		printf("\n3 - change Epoch length: %d", rSes.iEpochLength);
		printf("\n4 - change Samples used: %d", rSes.lSampleQtyReq);
		printf("\n5 X Synchronous commence");
		printf("\n6 - Asynchronous commence");
		printf("\n7 - change Blocks per kernel: %d", rSes.iBpropBlocks);
		printf("\n8 - change Threads per block: %d", rSes.iBpropThreads);
		printf("\n9 X ");
		printf("\n0 - Quit");
		CLIbase(rSes);
	}
	if(iMenuNum==90){
		printf("\n1 - convert .txt list of weights to .wgt");
		printf("\n2 X RMSE/evaluate test");
		printf("\n3 X classification test");
		printf("\n4 X backpropagation test (weights are restored)");
		printf("\n5 - Show CUDA properties ");
		printf("\n6 - change Epoch length: %d", rSes.iEpochLength);
		printf("\n7 - change Samples used: %d", rSes.lSampleQtyReq);
		printf("\n8 - change Blocks per kernel: %d", rSes.iBpropBlocks);
		printf("\n9 - change Threads per block: %d", rSes.iBpropThreads);
		printf("\n0 - Quit");
		CLIbase(rSes);
	}
	printf("\n");
	// http://www.cplusplus.com/doc/ascii/
	while(a<'0'||a>'9')
		a=_getch();
	return ((int)a)-48;
}


int cDrover::CLIbase(struct rohanContext& rSes)
{mIDfunc /// displays the base information common to each/most menus
		printf("\n %s %d samples MAX %f, %d trainable", rSes.rLearn->sLearnSet, rSes.rLearn->lSampleQty, rSes.dMAX, 
			TrainNNThresh(rSes, false, rSes.lSampleQtyReq));
		printf("\nRMSE: D %f, Y %f/%f ", rSes.dTargetRMSE, rSes.dHostRMSE, rSes.dDevRMSE);
		for(int i=0;i<rSes.rNet->iLayerQty;++i)
			printf("L%d %d; ", i, rSes.rNet->rLayer[i].iNeuronQty);
		printf("%d sectors ", rSes.rNet->iSectorQty);
		if (rSes.bRInJMode) printf("ReverseInput "); 
	return 1;
}

int BeginSession(struct rohanContext& rSes)
{mIDfunc /// accepts keyboard input to define the name of the session, which will be used to name certain output files.
	cout << "\nEnter a session name: ";
	cin >> rSes.sSesName; 

	return 1;
}


int cDrover::GetNNTop(struct rohanContext& rSes)
{mIDfunc /// sets up network poperties and data structures for use
	char sNeuronsPerLayer[254];
	int iSectorQty, iInputQty;

	cout << "Enter # of sectors (0 to return): ";
	cin >> iSectorQty;
	if(iSectorQty){
		cout << "Enter # of inputs (0 to return): ";
		cin >> iInputQty; // last chance to quit
	}
	if(iSectorQty && iInputQty) {
		Barge->cuFreeNNTop(rSes); // release old network structures
		rSes.rNet->iSectorQty=iSectorQty; // update sector qty
		rSes.rNet->kdiv2=iSectorQty/2; // update sector qty
		rSes.rLearn->iInputQty=iInputQty; // upsdate input qty
		cout << "Enter numbers of neurons per layer separated by commas, \ne.g. 63,18,1 : ";
		cin >> sNeuronsPerLayer;
		cuMakeLayers(iInputQty, sNeuronsPerLayer, rSes); // make new layers
		rSes.rNet->dK_DIV_TWO_PI = rSes.rNet->iSectorQty / TWO_PI; // Prevents redundant conversion operations
		cuMakeNNStructures(rSes); // allocates memory and populates network structural arrays
		cuRandomizeWeightsBlock(rSes); // populate newtork with random weight values
		printf("Random weights loaded.\n");
		printf("%d-valued logic sector table made.\n", cuSectorTableMake(rSes));
		printf("\n");
		return rSes.rNet->iLayerQty;
	}
	else
		return 999;
}

int cuMakeNNStructures(struct rohanContext &rSes)
{mIDfunc
/*! Initializes a neural network structure of the given number of layers and
 *  layer populations, allocates memory, and populates the set of weight values randomly.
 *
 * iLayerQty = 3 means Layer 1 and Layer 2 are "full" neurons, with output-only neurons on layer 0.
 * 0th neuron on each layer is a stub with no inputs and output is alawys 1+0i, to accomodate internal weights of next layer.
 * This allows values to be efficiently calculated by referring to all layers and neurons identically.
 * 
 * rLayer[1].iNeuronQty is # of neurons in Layer 1, not including 0
 * rLayer[2].iNeuronQty is # of neurons in Layer 2, not including 0
 * rLayer[0].iNeuronQty is # of inputs in Layer 0 
 * iNeuronQTY[1] is # of neurons in Layer 1, including 0
 * iNeuronQTY[2] is # of neurons in Layer 2, including 0 */

	long lReturn=0;
//const cuDoubleComplex cdcZero = { 0, 0 }, 
	const cuDoubleComplex cdcInit = { -999.0, 999.0 };
	//cdcInit.x=-999.0; cdcInit.y=999.0;
	for (int i=0; i < rSes.rNet->iLayerQty; ++i){  //Layer Zero has no need of weights! 8/13/2010
		struct rohanLayer& lay = rSes.rNet->rLayer[i];
		struct rohanNetwork * rnSrc = rSes.rNet;
		long DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE, L=i;
		//setup dimension values
		DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
		DSIZE = DQTY * sizeof(cuDoubleComplex) ;
		NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
		NSIZE = NQTY * sizeof(cuDoubleComplex) ;
		WQTY = DQTY * NQTY ; // weights = neurons * dendrites
		WSIZE = WQTY * sizeof(cuDoubleComplex) ;
		
		//allocate memory
		lay.Weights = (cuDoubleComplex*)malloc ( WSIZE ); // 2D array of complex weights
			mCheckMallocWorked(lay.Weights)
		lay.XInputs = (cuDoubleComplex*)malloc( DSIZE ); //allocate a pointer to an array of outputs
			mCheckMallocWorked(lay.XInputs)
		lay.ZOutputs = (cuDoubleComplex*)malloc( NSIZE ); //allocate a pointer to an array of outputs
			mCheckMallocWorked(lay.ZOutputs)
		lay.Deltas = (cuDoubleComplex*)malloc( NSIZE ); //allocate a pointer to a parallel array of learned corrections
			mCheckMallocWorked(lay.Deltas)
		lReturn+=lay.iNeuronQty*lay.iDendriteQty;
   		lReturn+=lay.iNeuronQty;
	
		//init values
		for (int i=0; i <= lay.iDendriteQty; ++i){
			for (int k=0; k <= lay.iNeuronQty; ++k){ 
				lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)].x=(double)rand()/65535; // necessary to promote one operand to double to get a double result
				lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)].y=(double)rand()/65535;
				//lay.Deltas[IDX2C(i, k, lay.iDendriteQty+1)]=cdcInit;
			}
			// reset neuron 0 weights to null
			lay.Weights[IDX2C(i, 0, lay.iDendriteQty+1)] = cdcZero;
			// mark inputs as yet-unused
			lay.XInputs[i]=cdcInit;
		}
		lay.Weights[IDX2C(0, 0, lay.iDendriteQty+1)].x=1.0; // neuron 0, dendrite 0 interior weight should always be equal to 1+0i
		for (int k=0; k <= lay.iNeuronQty; ++k){
			// mark outputs and deltas as yet-unused
			lay.ZOutputs[k]=cdcInit;
			lay.Deltas[k]=cdcInit;
		}
	}
	return lReturn; //return how many weights and outputs allocated
}


int cDrover::GetWeightSet(struct rohanContext& rSes)
{mIDfunc /// chooses and loads the weight set to be worked with
	int iReturn=0; 
	char sWeightSet[254];
	FILE *fileInput;
	
	cout << "Enter name of binary weight set: ";
	std::cin.clear();
	std::cin >> sWeightSet;
	strcat(sWeightSet, ".wgt");

	// File handle for input
	iReturn=BinaryFileHandleRead(sWeightSet, &fileInput);
	if (iReturn==0) // unable to open file
		++rSes.iErrors;
	else{ // file opened normally
		// file opening and reading are separated to allow for streams to be added later
		iReturn=cuNNLoadWeights(rSes, fileInput); // reads weights into layered structures
		if (iReturn) {
			printf("%d weights read.\n", iReturn);
			Barge->LayersToBlocks(rSes); //, *rSes.rNet);
		}
		else {
			printf("No Weights Read\n");
			iReturn=0;
		}
	}
	printf("\n");
	return iReturn;
}


long cDrover::LetInteractiveEvaluation(struct rohanContext& rSes)
{mIDfunc /// allows user to ask for different number of samples to be evaluated
	int iReturn=0, iSelect=1;
	
	while(iSelect){
		//Team->LetSlack(rSes);
		//Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
		//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
		iSelect=DisplayMenu(50, rSes);
		if (iSelect==1) {rSes.iSaveInputs=(rSes.iSaveInputs ? false: true); }
		if (iSelect==2) {rSes.iSaveOutputs=(rSes.iSaveOutputs ? false: true); }
		if (iSelect==3) {rSes.iSaveSampleIndex=(rSes.iSaveSampleIndex ? false: true); }
		if (iSelect==4) {printf("Enter samples requested\n");std::cin >> rSes.lSampleQtyReq;} //
		if (iSelect==5) { // serial values are computed and then displayed
					++iReturn; 
					boost::timer::auto_cpu_timer t;
					cuEvalNNLearnSet(rSes, rSes.lSampleQtyReq);
					RmseNN(rSes, rSes.lSampleQtyReq);
					printf("%s: first %d samples requested\nRMSE= %f", rSes.rNet->sWeightSet, rSes.lSampleQtyReq, rSes.dHostRMSE);		
		}
		if (iSelect==6) { // asynchronous kernel launch
					++iReturn;
					// device values are computed and then displayed
					Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'D'); // eval on device
					Team->GetRmseNN(rSes, rSes.lSampleQtyReq);
					printf("%s: first %d samples requested\nRMSE= %f", rSes.rNet->sWeightSet, rSes.lSampleQtyReq, rSes.dDevRMSE);		
		}
		if (iSelect==7) {printf("Enter blocks per kernel\n");std::cin >> rSes.iEvalBlocks;}
		if (iSelect==8) {printf("Enter threads per block\n");std::cin >> rSes.iEvalThreads;}
		if (iSelect==9) {Barge->LetWriteEvals(rSes, *rSes.rLearn);} 
		if (iSelect==0) {} // quit
	}
	return iReturn;
}


long cDrover::LetInteractiveLearning(struct rohanContext& rSes)
{mIDfunc /// allows user to select learning thresholds
	int iReturn=0, iSelect=1;
	
	while(iSelect){
		//Team->LetSlack(rSes);
		//Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
		//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
		iSelect=DisplayMenu(60, rSes);
		if (iSelect==1) {printf("Enter desired RMSE for learning\n");std::cin >> rSes.dTargetRMSE;}
		if (iSelect==2) {printf("Enter MAX allowable error per sample\n");std::cin >> rSes.dMAX;}
		if (iSelect==3) {printf("Enter iterations per epoch\n");std::cin >> rSes.iEpochLength;}
		if (iSelect==4) {printf("Enter samples requested\n");std::cin >> rSes.lSampleQtyReq;} //
		//if (iSelect==5) {} // synchronous kernel launch
		if (iSelect==6) { // asynchronous kernel launch
					++iReturn;
							Team->LetTaut(rSes);
							//rSes.lSamplesTrainable=knlBackProp( rSes, rSes.lSampleQtyReq, 1, 'R');
							rSes.lSamplesTrainable=Team->LetTrainNNThresh( rSes, rSes.lSampleQtyReq, 1, 'R', rSes.dTargetRMSE, rSes.iEpochLength);
		}
		if (iSelect==7) {printf("Enter blocks per kernel\n");std::cin >> rSes.iBpropBlocks;}
		if (iSelect==8) {printf("Enter threads per block\n");std::cin >> rSes.iBpropThreads;}
		//if (iSelect==9) {} //
		if (iSelect==0) {} // quit
	}
	return iReturn;
}


long cDrover::LetUtilities(struct rohanContext& rSes)
{mIDfunc /// allows user to select learning thresholds
	int iReturn=0, iSelect=1;
	int iEpoch=rSes.iEpochLength;
	int iSampleQtyReq=rSes.lSampleQtyReq;
	int iBpropBlocks=rSes.iBpropBlocks;
	int iBpropThreads=rSes.iBpropThreads;
	
	while(iSelect){
		iSelect=DisplayMenu(90, rSes);
		if (iSelect==1) {
			char afname[100]="", bfname[100]="", 
				lineIn[255], *tok, *cSample;
			FILE * ASCIN, * BINOUT; cuDoubleComplex way;
			cout << "Enter name of .txt file to convert to .wgt:" << endl;
			cin >> bfname;
			strcat(afname, bfname); strcat(afname, ".txt"); strcat(bfname, ".wgt");
			AsciiFileHandleRead( afname, &ASCIN );
			BinaryFileHandleWrite( bfname, &BINOUT );
			#define MAX_REC_LEN 65536 /* Maximum size of input buffer */
			while(fgets(lineIn, MAX_REC_LEN, ASCIN)) { //each line is read in turn
				cSample = _strdup(lineIn);
				printf("%s", cSample); 
				tok=strtok( cSample , " ,\t" ); way.x=atof( tok ); 
				tok=strtok( NULL, " ,\t" ); way.y=atof( tok );
				printf("%f+%f\n", way.x, way.y);
				fwrite( &(way.x) , sizeof(double), 1, BINOUT);
				fwrite( &(way.y) , sizeof(double), 1, BINOUT);
			}
			fclose(ASCIN);
			fclose(BINOUT);
		}
		if (iSelect==2) {printf("Enter MAX allowable error per sample\n");std::cin >> rSes.dMAX;}
		if (iSelect==5) {Team->CUDAShowProperties(rSes, rSes.iMasterCalcHw, stdout);}
		if (iSelect==6) {printf("Enter iterations per epoch\n");std::cin >> rSes.iEpochLength;}
		if (iSelect==7) {printf("Enter samples requested\n");std::cin >> rSes.lSampleQtyReq;} //
		if (iSelect==8) {printf("Enter blocks per kernel\n");std::cin >> rSes.iBpropBlocks;}
		if (iSelect==9) {printf("Enter threads per block\n");std::cin >> rSes.iBpropThreads;}
		if (iSelect==0) {} // quit
	}
	return iReturn;
}


void cDrover::RLog(struct rohanContext& rSes, char * sLogEntry)
{mIDfunc // logs strings describing events, preceeded by the local time
	using namespace boost::posix_time; 
    ptime now = second_clock::local_time(); //use the clock 
    sLogEntry=strtok(sLogEntry, "\n"); // trim any trailing chars
	*(rSes.ofsRLog) << now << " " << sLogEntry  << endl;
}


long cDrover::DoPostLoop(struct rohanContext& rSes) 
{mIDfunc /// Final operations including freeing of dynamically allocated memory are called from here. 
	int iReturn=0, iSelect=0;

	DoEndItAll(rSes);
	printf("Program terminated after %d warning(s), %d operational error(s).\n", rSes.iWarnings, rSes.iErrors);
	printf("Waiting on keystroke...\n");
	_getch();

	return 0;
}


long cDrover::DoEndItAll(struct rohanContext& rSes)
{mIDfunc /// prepares for graceful ending of program
	int iReturn=0;

	Team->LetUnHitch(rSes);
	iReturn=Barge->DoCuFree(rSes);
	
	return iReturn;
}

