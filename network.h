#ifndef NETWORK_H
#define NETWORK_H

typedef struct Net {
    int hiddenNum; //Number of hidden layers
    int hiddenSize; //Number of neurons in each hidden layer
    int inputNum; //Number of neurons in input layer
    int outputNum; //Number of neurons in output layer
    int sigmoid; //1 indicates sigmoid
    int relu; //1 indicates relu
    double ***weights; //Array of matrices of the weights connecting each layer
    double **hiddenZ; //Matrix containing the hidden neurons' pre-activation values. Used in backpropagation
    double ** hidden;  //Matrix contain the hidden neurons' post-activation values
    double **biases; //Matrix that contains the biases for each layer of neurons
    double *inputs; //Array containing the input neurons' values
    double *outputsZ; //Array containing the output neurons' pre-activation values. Used in backpropagation
    double* outputs; //Array containing the output neurons' post-activation values
} Net;

typedef struct Data { //Struct to keep track of data from .csv
    int rowCount;
    int columnCount;
    double** data;
} Data;

Net open(char* fileName);

void deleteNet(Net *net);

void save(Net *net, char *fileName);

void compute(Net *net);

Net createNetwork(int inputNeurons, int outputNeurons, int hiddenLayers, int hiddenSize, char* sigmoidOrRelu);

void backProp(Net* net, Data* trainData, int batches, int batchSize, double learnRate);

void testNetwork(Net* net, Data* testData);

#endif
