long LoadNNWeights(int iLayerQty, int iNeuronQty[], double ****dWeightsR, double ****dWeightsI, FILE *fileInput){
// pulls in values from .wgt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	long lReturnValue=0;
	for (int i=1; i < iLayerQty; ++i){ //no weights for layer 0
		for (int j=1; j <= iNeuronQty[i]; ++j){ // no weights for neuron 0
			for (int k=0; k <= iNeuronQty[i-1]; ++k){
				fread(&(*dWeightsR)[i][j][k], sizeof(double), 1, fileInput);
				fread(&(*dWeightsI)[i][j][k], sizeof(double), 1, fileInput);
				lReturnValue++;
				//printf("weight %d.%d.%d= % 1f + % 1f i\n",  i, j, k, (*dWeightsR)[i][j][k], (*dWeightsI)[i][j][k]);
			}
		}
	}
	fclose(fileInput);
	return lReturnValue;
}

long EvalNNSamplesCous(int iLayerQty, int *iNeuronQty, double ***dWeightsR, double ***dWeightsI, double **dOutputsR, 
					   double **dOutputsI, int iSectorQty, double *dX, double *dY, long lSampleQty, int iValuesPerLine, 
					   int **iValuesLearn, int ***iEvalResult)
{
// This will apply a MLMVN weight set to each sample of a learning set in turn and record the resulting final output for each.
// Discrete inputs and outputs are used. Real integers are convered via K-valued logic to complex coordinates,
// which are then product-summed by successive layers of neurons, then conveted back to integer output
	long s;
	// sample index, counts up
	//
	//(*iEvalResult)=(int**)malloc((1+iNeuronQty[iLayerQty-1]) * sizeof(int*)); // point to an array of array of ints
	//(*iEvalResult)[1]=(int*)malloc(lSampleQty * sizeof(int)); // point to an array of ints
	//	if ((*iEvalResult==NULL) || ((*iEvalResult)[0]==NULL)) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
	
	// here beginneth ye main duty loop
	for (s=0; s<lSampleQty; ++s){
// layer zero (inputs) is special
		for (int i=0; i<iNeuronQty[0]; ++i){
			dOutputsR[0][i]=dX[iValuesLearn[s][i]]; // virtual input neurons' output is inputs converted to complex coords
			dOutputsI[0][i]=dY[iValuesLearn[s][i]]; // layer 0, ith neuron
			//printf("%d-%d: %.2f %.2f %d\n", s, i, dXmiddle[iValuesLearn[s][i]], dYmiddle[iValuesLearn[s][i]], iValuesLearn[s][i]);
			//printf("s%d, L0, n%d: % 1.2f + % 1.2f i (%d)\n", s, i, dOutputsR[0][i], dOutputsI[0][i], iValuesLearn[s][i]);
		}
// middle and top layers
		int iLastLayer=iLayerQty-1;
		for (int L=1; L<=iLastLayer; ++L){
			int iLastNeuron=iNeuronQty[L]; // size of current layer
			int PL=L-1; // index of previous layer
			int iLastWeight=iNeuronQty[PL]; // weight qty depends on size of previous layer
			for (int i=1; i<=iLastNeuron; ++i){ //Neuron zero is skipped to leave its output at 1+0i
				dOutputsR[L][i]=dWeightsR[L][i][0];
				dOutputsI[L][i]=dWeightsI[L][i][0];
					//if (s==0) printf("s%d, L%d, n%d, j%d: %.8f + %.8f i\n", s, L, i, 0, dWeightsR[L][i][0], dWeightsI[L][i][0]);
				for (int j=1; j<=iLastWeight; ++j){ //walk weights on inputs from lower layer
					dOutputsR[L][i] += (dWeightsR[L][i][j]*dOutputsR[PL][j] - dWeightsI[L][i][j]*dOutputsI[PL][j]); // accumulate first and last
					dOutputsI[L][i] += (dWeightsI[L][i][j]*dOutputsR[PL][j] + dWeightsR[L][i][j]*dOutputsI[PL][j]);	// accumulate inside and outside			
					//if (s==0) printf("s%d, L%d, n%d, j%d: %.8f + %.8f i\n", s, L, i, j, dWeightsR[L][i][j], dWeightsI[L][i][j]);
				}
				//printf("s%d, L%d, n%d Sigma: % 1.2f + % 1.2f i\n", s, L, i, dOutputsR[L][i], dOutputsI[L][i]);
				// apply Continuous activation function to weighted sum : phi(z)=z/|z|
				double A=dOutputsR[L][i]; double B=dOutputsI[L][i];
				double C=sqrt(A*A+B*B);
				dOutputsR[L][i]=dOutputsR[L][i]/C;
				dOutputsI[L][i]=dOutputsI[L][i]/C;
				
				// apply Discrete activation function to weighted sum : s=int(arctan(z)*k/2pi), phi(z)=(X(s),Y(s))
				//double theta = atan2(dOutputsI[L][i], dOutputsR[L][i]); // theta = arctan y/x
				//int iS = (int)((iSectorQty * theta / TWO_PI) + iSectorQty) % iSectorQty;
				//dOutputsR[L][i]=(dX)[iS];
				//dOutputsI[L][i]=(dY)[iS];
				
				//printf("s%d, L%d, n%d output: % 1.2f + % 1.2f i\n", s, L, i, dOutputsR[L][i], dOutputsI[L][i]);
			}
		}
// last layer is also special
		double K_DIV_TWO_PI = iSectorQty / TWO_PI; // Prevents redundant conversion operations
		for (int i=1; i<=iNeuronQty[iLastLayer];i++){
			double theta = atan2(dOutputsI[iLastLayer][i], dOutputsR[iLastLayer][i]); // theta = arctan y/x
			//(*iEvalResult)[i][s]= (int)((iSectorQty * theta / TWO_PI) + iSectorQty) % iSectorQty;
			(*iEvalResult)[i][s]= (int)((theta * K_DIV_TWO_PI) + iSectorQty) % iSectorQty;
			//printf("s%d, L%d, n%d: % 1.2f theta, %d sector\n", s, iLayerQty-1, i, theta, iResultSet[i][s]);
		}
		// end of main duty loop, go back for the next sample
	}
	return s; // return qty samples evaluated
}


int MakeNNStructures(int iLayerQty, int *iNeuronQty, double ****dWeightsR, double ****dWeightsI, 
					 double ***dOutputsR, double ***dOutputsI)
{
// Initializes a neural network structure of the given number of layers and
// layer populations, allocates memory, and populates the set of weight values randomly.
//
// iLayerQty = 3 means Layer 1 and Layer 2 are "full" neurons, with output-only neurons on layer 0.
// 0th neuron on each layer is a stub with no inputs and output is alawys 1+0i, to accomodate internal weights of next layer.
// This allows values to be efficiently calculated by referring to all layers and neurons identically.

// iNeuronQty[1] is # of neurons in Layer 1
// iNeuronQty[2] is # of neurons in Layer 2
// iNeuronQty[0] is # of inputs in Layer 0
	int iUnderLayer; long lReturn=0;
	
	// real weights
	(*dWeightsR) = (double***)malloc(iLayerQty * sizeof (double**)); //allocate one pointer to an array of arrays of arrays of weights
		if (*dWeightsR==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
	for (int i=1; i < iLayerQty; i++){ //Layer Zero has no need of weights!
		(*dWeightsR)[i] = (double**)malloc((iNeuronQty[i]+1) * sizeof (double*)); //allocate a pointer to an array of of arrays of weights
			if ((*dWeightsR)[i]==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
		iUnderLayer=i-1;
		lReturn+=(iNeuronQty[i]*(iNeuronQty[iUnderLayer]+1));
		for (int j=1; j <= iNeuronQty[i]; j++){ //Neuron Zero has no need of weights!
			(*dWeightsR)[i][j] = (double*)malloc((iNeuronQty[iUnderLayer]+1) * sizeof (double)); //allocate a pointer to an array of weights
				if ((*dWeightsR)[i][j]==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
			for (int k=0; k <= iNeuronQty[iUnderLayer]; k++){
				(*dWeightsR)[i][j][k]=(double)rand()/65535; // necessary to promote one operand to double to get a double result
			}
		}
	}

	// imaginary weights
	(*dWeightsI) = (double***)malloc(iLayerQty * sizeof (double**)); //allocate one pointer to an array of arrays of arrays of weights
		if (*dWeightsI==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
	for (int i=1; i < iLayerQty; i++){ //Layer Zero has no need of weights!
		(*dWeightsI)[i] = (double**)malloc((iNeuronQty[i]+1) * sizeof (double*)); //allocate a pointer to an array of of arrays of weights
			if ((*dWeightsI)[i]==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
		iUnderLayer=i-1;
		lReturn+=(iNeuronQty[i]*(iNeuronQty[iUnderLayer]+1));
		for (int j=1; j <= iNeuronQty[i]; j++){ //Neuron Zero has no need of weights!
			(*dWeightsI)[i][j] = (double*)malloc((iNeuronQty[iUnderLayer]+1) * sizeof (double)); //allocate a pointer to an array of weights
				if ((*dWeightsI)[i][j]==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
			for (int k=0; k <= iNeuronQty[iUnderLayer]; k++){
				(*dWeightsI)[i][j][k]=(double)rand()/65535; // necessary to promote one operand to double to get a double result
			}
		}
	}
	//for (int i=1; i < iLayerQty; i++){
	//	for (int j=1; j <= iNeuronQty[i]; j++){
	//		for (int k=0; k <= iNeuronQty[i-1]; k++){
	//			printf("InputWt %d.%d.%d==%.2f + %.2fi\n",  i, j, k, (*dWeightsR)[i][j][k],(*dWeightsI)[i][j][k]);
	//		}
	//		printf("Waiting on keystroke...\n");
	//		_getch(); // wait on keystroke
	//		}
	//}

	// real outputs
	(*dOutputsR) = (double**)malloc(iLayerQty * sizeof (double*)); //allocate one pointer to an array of arrays of outputs
		if ((*dOutputsR)==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
	for (int i=0; i < iLayerQty; i++){
		(*dOutputsR)[i] = (double*)malloc((iNeuronQty[i]+1) * sizeof (double)); //allocate a pointer to an array of of arrays of weights
			if ((*dOutputsR)[i]==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
		lReturn+=iNeuronQty[i];
		for (int j=0; j <= iNeuronQty[i]; j++){
			(*dOutputsR)[i][j]=1;
			//(*dOutputsR)[i][j]=i+(double)j/100; // necessary to promote one operand to double to get a double result
		}
	}

	// imaginary outputs
	(*dOutputsI) = (double**)malloc(iLayerQty * sizeof (double*)); //allocate one pointer to an array of arrays of outputs
		if ((*dOutputsI)==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
	for (int i=0; i < iLayerQty; i++){
		(*dOutputsI)[i] = (double*)malloc((iNeuronQty[i]+1) * sizeof (double)); //allocate a pointer to an array of of arrays of weights
			if ((*dOutputsI)[i]==NULL) return (0 * printf("Malloc ran out of space?  OH NOES!\n"));
		lReturn+=iNeuronQty[i];
		for (int j=0; j <= iNeuronQty[i]; j++){
			(*dOutputsI)[i][j]=0;
			//(*dOutputsI)[i][j]=i+(double)j/100; // necessary to promote one operand to double to get a double result
		}
	}

	//for (int i=0; i < iLayerQty; i++){
	//	//printf("Layer %d has %d neurons each with 1 outputs.\n", i, iNeuronQty[i]);
	//	for (int j=0; j <= iNeuronQty[i]; j++){
	//		printf("Neuron %d.%d=%.2f + %.2fi\n",  i, j,(*dOutputsR)[i][j], (*dOutputsI)[i][j]);
	//	}
	//}

	return lReturn; //return how many weights and outputs allocated
}

