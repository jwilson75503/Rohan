/* Includes, cuda */

int BinaryFileHandleRead(char* sFileName, FILE** fileInput)
;
long BinaryFileHandleWrite(char *sFileName, FILE **fileOutput)
;
int AsciiFileHandleRead(char *sFileName, FILE **fileInput)
;
int AsciiFileHandleWrite(char *sFilePath, char *sFileName, FILE **fileOutput)
;
long AsciiWeightDump(struct rohanContext& rSes, FILE *fileOutput)
;
long LoadNNWeights(int iLayerQty, int iNeuronQty[], double ****dWeightsR, double ****dWeightsI, FILE *fileInput)
;
int cuMessage(cublasStatus csStatus, char *sName, char *sCodeFile, int iLine, char *sFunc)
;
int cuMakeLayers(int iInputQty, char *sLayerSizes, struct rohanContext& rSes)
;
int cuMakeArchValues(char *sMLMVNarch, struct rohanContext& rSes)
;
long cuLoadSampleSet(struct rohanContext& rSes, FILE *fileInput)
;
long cuReLoadSampleSet(struct rohanContext& rSes, FILE *fileInput)
;
long cuNNLoadWeights(struct rohanContext& rSes, FILE *fileInput)
;
long cuPreSaveNNWeights(struct rohanContext& rSes)
;
long cuSaveNNWeights(struct rohanContext& rSes, FILE *fileOutput)
;
long cuSaveNNWeightsASCII(struct rohanContext& rSes, FILE *fileOutput)
;
int devCopyArchValues(struct rohanContext& rSes)
;
long devCopySampleSet(struct rohanContext& rSes)
;
long devCopyNNWeights(struct rohanContext& rSes)
;
int devPrepareNetwork(struct rohanContext& rSes)
;
