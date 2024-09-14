#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "network.h"
#include <math.h>
#include <time.h>
#include "data.h"

double function(int x) {
    return sin(x) + 2;
}

double derivative(Net *net, double x) {
    if(net->relu == 1) { //****
       if(x > 0) return 1;
       return 0;
    }
    if(net->sigmoid == 1) {
        return exp(x)/((exp(x) + 1) * (exp(x) + 1));
    }
    return 1; //Linear activation
}

double relu(double val) {
    if(val > 0) return val;
    return 0;
}

double sigmoid(double val) {
    //return val;
    return 1/(1+exp(-val));
}

double*** createWeights(Net* net) {
    double*** weights = (double***)malloc((net->hiddenNum+1)*sizeof(double**)); //Allocates memory for array of matrices for storing weights
    if(net->hiddenNum == 0) { //Checks if no hidden layers were specified
        weights[0] = (double**)malloc(net->inputNum*sizeof(double*)); //Allocates memory for matrix
        for(int j = 0; j<net->inputNum; j++) { 
            weights[0][j] = (double*)calloc(net->outputNum, sizeof(double)); //Allocates memory for each row in matrix
        }
        //printf("wow");
    }
    else {
        for(int i = 0; i<net->hiddenNum+1; i++) { //Iterates over array of weight matrices to allocate memory for each matrix
            if(i == 0) { //Checks if it is the first layer: input layer -> hidden layer
                weights[i] = (double**)malloc(net->inputNum*sizeof(double*)); //Allocates memory for matrix (array of pointers)
                for(int j = 0; j<net->inputNum; j++) {
                    weights[i][j] = (double*)calloc(net->hiddenSize, sizeof(double)); //Allocates memory for a row (array of values)
                }
            }
            else if(i == net->hiddenNum) { //Checks if it is the last layer: hidden layer -> output layer
                weights[i] = (double**)malloc(net->hiddenSize*sizeof(double*)); //Allocates memory for matrix (array of pointers)
                for(int j = 0; j<net->hiddenSize; j++) {
                    weights[i][j] = (double*)calloc(net->outputNum, sizeof(double)); //Allocates memory for a row (array of values)
                }
            }
            else { //Hidden layer -> hidden layer
                weights[i] = (double**)malloc(net->hiddenSize*sizeof(double*)); //Allocates memory for matrix (array of pointers)
                for(int j = 0; j<net->hiddenSize; j++) {
                    weights[i][j] = (double*)calloc(net->hiddenSize, sizeof(double)); //Allocates memory for a row (array of values)
                }
            }
        }
    }
    return weights;
}

double** createBias(Net* net) {
    double** biases = (double**)malloc((net->hiddenNum+1)*sizeof(double*));

    for(int j = 0; j<net->hiddenNum+1; j++) { //Allocates memory for each layer
        if(net->hiddenNum == 0) {
            biases[j] = (double*)calloc(net->outputNum, sizeof(double));
        }
        else if(j == 0) {
            biases[j] = (double*)calloc(net->hiddenSize, sizeof(double));
        }
        else if(j == net->hiddenNum) {
            biases[j] = (double*)calloc(net->outputNum, sizeof(double));
        }
        else {
            biases[j] = (double*)calloc(net->hiddenSize, sizeof(double));
        }
    }
    return biases;
}

double** createAct(Net *net) {
    if(net->hiddenNum == 0) {
        return NULL;
    }
    double **actDeriv = (double**)malloc(sizeof(double*) * net->hiddenNum);
    for(int i = 0; i < net->hiddenNum; i++) {
        actDeriv[i] = (double*)calloc(net->hiddenSize, sizeof(double));
    }
    return actDeriv;
}

Net open(char* fileName) {
    FILE *fp = fopen(fileName, "rb");
    Net net;
    fread(&net.hiddenNum, sizeof(int), 1, fp);
    fread(&net.hiddenSize, sizeof(int), 1, fp);
    fread(&net.inputNum, sizeof(int), 1, fp);
    fread(&net.outputNum, sizeof(int), 1, fp);
    fread(&net.sigmoid, sizeof(int), 1, fp);
    fread(&net.relu, sizeof(int), 1, fp);
    int hiddenLayers = net.hiddenNum;
    int hiddenSize = net.hiddenSize;
    int inputNeurons = net.inputNum;
    int outputNeurons = net.outputNum;
    //printf("%d\n", outputNeurons);
    net.inputs = (double*)malloc(inputNeurons*sizeof(double)); //Allocates memory for the input layer
    net.hidden = (double**)malloc(hiddenLayers*sizeof(double*));
    net.hiddenZ = (double**)malloc(hiddenLayers*sizeof(double*));
    for(int j = 0; j < hiddenLayers; j++) {
        net.hidden[j] = (double*)malloc(hiddenSize*sizeof(double));
        net.hiddenZ[j] = (double*)malloc(hiddenSize*sizeof(double));
    }
    net.outputs = (double*)malloc(outputNeurons*sizeof(double)); //allocates memory for output layer
    net.outputsZ = (double*)malloc(outputNeurons*sizeof(double));
    net.weights = createWeights(&net);
    net.biases = createBias(&net);
    for(int matrix = 0; matrix < net.hiddenNum + 1; matrix++) { //Reads all the weights from the file
        if(net.hiddenNum == 0) { //Input -> output
            for(int j = 0; j < net.inputNum; j++) {
                fread(net.weights[matrix][j], sizeof(double), net.outputNum, fp);
            }
        }
        else if(matrix == 0) { //Input -> hidden layer
            for(int j = 0; j < net.inputNum; j++) {
                fread(net.weights[matrix][j], sizeof(double), net.hiddenSize, fp);
            }
        }
        else if(matrix == net.hiddenNum) { //hidden layer -> output 
            for(int j = 0; j < net.hiddenSize; j++) {
                fread(net.weights[matrix][j], sizeof(double), net.outputNum, fp);
            }
        }
        else { //hidden layer -> hidden layer
            for(int j = 0; j < net.hiddenSize; j++) {
                fread(net.weights[matrix][j], sizeof(double), net.hiddenSize, fp);
            }
        }
    }
    for(int j = 0; j < net.hiddenNum + 1; j++) { //Reads all the bias data from the file
        if(net.hiddenNum == 0) { //output layer biases
            fread(net.biases[j], sizeof(double), net.outputNum, fp);
        }
        else if(j == 0) { //hidden layer biases
            fread(net.biases[j], sizeof(double), net.hiddenSize, fp);
        }
        else if(j == net.hiddenNum) { //output layer biases
            fread(net.biases[j], sizeof(double), net.outputNum, fp);
        }
        else { //Hidden layer biases
            fread(net.biases[j], sizeof(double), net.hiddenSize, fp);
        }
    }
    fclose(fp);
    return net;
}

void deleteNet(Net *net) {
    for(int matrix = 0; matrix < net->hiddenNum + 1; matrix++) { //Frees all the memory used by the weights
        if(net->hiddenNum == 0) { //Input -> output
            for(int j = 0; j < net->inputNum; j++) {
                free(net->weights[matrix][j]);
            }
        }
        else if(matrix == 0) { //Input -> hidden layer
            for(int j = 0; j < net->inputNum; j++) {
                free(net->weights[matrix][j]);
            }
        }
        else if(matrix == net->hiddenNum) { //hidden layer -> output 
            for(int j = 0; j < net->hiddenSize; j++) {
                free(net->weights[matrix][j]);
            }
        }
        else { //hidden layer -> hidden layer
            for(int j = 0; j < net->hiddenSize; j++) {
                free(net->weights[matrix][j]);
            }
        }
        free(net->weights[matrix]);
    }
    free(net->weights);

    if(net->hiddenNum != 0) { //Frees all the memory used by the hidden layers
        for(int j = 0; j < net->hiddenNum; j++) {
            free(net->hidden[j]);
        }
        free(net->hidden);
    }
    
    for(int j = 0; j < net->hiddenNum + 1; j++) { //Frees all the memory used by the biases. 
        if(net->hiddenNum == 0) { //output layer biases
            free(net->biases[j]);
        }
        else if(j == 0) { //hidden layer biases
            free(net->biases[j]);
        }
        else if(j == net->hiddenNum) { //output layer biases
            free(net->biases[j]);
        }
        else { //Hidden layer biases
            free(net->biases[j]);
        }
    }
    free(net->biases);
    free(net->inputs);
    free(net->outputs);    
}

void save(Net *net, char *fileName) {
    FILE *fp = fopen(fileName, "wb");
    fwrite(&net->hiddenNum, sizeof(int), 1, fp);
    fwrite(&net->hiddenSize, sizeof(int), 1, fp);
    fwrite(&net->inputNum, sizeof(int), 1, fp);
    fwrite(&net->outputNum, sizeof(int), 1, fp);
    fwrite(&net->sigmoid, sizeof(int), 1, fp);
    fwrite(&net->relu, sizeof(int), 1, fp);
    for(int matrix = 0; matrix < net->hiddenNum + 1; matrix++) { //Writes all the weights to the file
        if(net->hiddenNum == 0) { //Input -> output
            for(int j = 0; j < net->inputNum; j++) {
                fwrite(net->weights[matrix][j], sizeof(double), net->outputNum, fp);
            }
        }
        else if(matrix == 0) { //Input -> hidden layer
            for(int j = 0; j < net->inputNum; j++) {
                fwrite(net->weights[matrix][j], sizeof(double), net->hiddenSize, fp);
            }
        }
        else if(matrix == net->hiddenNum) { //hidden layer -> output 
            for(int j = 0; j < net->hiddenSize; j++) {
                fwrite(net->weights[matrix][j], sizeof(double), net->outputNum, fp);
            }
        }
        else { //hidden layer -> hidden layer
            for(int j = 0; j < net->hiddenSize; j++) {
                fwrite(net->weights[matrix][j], sizeof(double), net->hiddenSize, fp);
            }
        }
    }

    for(int j = 0; j < net->hiddenNum + 1; j++) { //Writes all the bias data to the file
        if(net->hiddenNum == 0) { //output layer biases
            fwrite(net->biases[j], sizeof(double), net->outputNum, fp);
        }
        else if(j == 0) { //hidden layer biases
            fwrite(net->biases[j], sizeof(double), net->hiddenSize, fp);
        }
        else if(j == net->hiddenNum) { //output layer biases
            fwrite(net->biases[j], sizeof(double), net->outputNum, fp);
        }
        else { //Hidden layer biases
            fwrite(net->biases[j], sizeof(double), net->hiddenSize, fp);
        }
    }
    fclose(fp);    
    return;
}

void compute(Net *net) {
    double tmp = 0; //Used to temporarily store calculation results
    for(int matrix = 0; matrix < net->hiddenNum + 1; matrix++) { //Iterates through array of matrices to compute the neural network
        if(net->hiddenNum == 0) { //Checks if there are no hidden layers
            for(int i = 0; i < net->outputNum; i++) { //
                for(int j = 0; j < net->inputNum; j++) { 
                    tmp = tmp + net->weights[matrix][j][i]*net->inputs[j];
                }
                if(net->sigmoid == 1) {
                    net->outputs[i] = sigmoid(tmp + net->biases[matrix][i]);
                }
                else if(net->relu == 1) {
                    net->outputs[i] = relu(tmp + net->biases[matrix][i]);
                }
                else {
                    net->outputs[i] = tmp + net->biases[matrix][i];
                }
                //net->outputs[i] = sigmoid(tmp + net->biases[matrix][i]); 
                net->outputs[i] = tmp + net->biases[matrix][i]; //sigmoid(tmp + net->biases[matrix][i]); ****Temporary change for 2x
                net->outputsZ[i] = tmp + net->biases[matrix][i];
                tmp = 0;
            }
        }
        else if(matrix == 0) {
            for(int i = 0; i < net->hiddenSize; i++) {
                for(int j = 0; j < net->inputNum; j++) {
                    tmp = tmp + net->weights[matrix][j][i]*net->inputs[j];
                }
                if(net->sigmoid == 1) {
                    net->hidden[0][i] = sigmoid(tmp + net->biases[matrix][i]);
                }
                else if(net->relu == 1) {
                    net->hidden[0][i] = relu(tmp + net->biases[matrix][i]);
                }
                else {
                    net->hidden[0][i] = tmp + net->biases[matrix][i];
                }
                net->hiddenZ[0][i] = tmp + net->biases[matrix][i]; //pre-activation values to be used in backpropagation
                //net->hidden[0][i] = sigmoid(tmp + net->biases[0][i]);
                tmp = 0;
            }
        }
        else if(matrix == net->hiddenNum) {
            for(int i = 0; i < net->outputNum; i++) {
                for(int j = 0; j < net->hiddenSize; j++) {
                    tmp = tmp + net->weights[matrix][j][i]*net->hidden[net->hiddenNum-1][j];
                }
                if(net->sigmoid == 1) {
                    net->outputs[i] = sigmoid(tmp + net->biases[matrix][i]);
                }
                else if(net->relu == 1) {
                    net->outputs[i] = relu(tmp + net->biases[matrix][i]);
                }
                else {
                    net->outputs[i] = tmp + net->biases[matrix][i];
                }
                net->outputsZ[i] = tmp + net->biases[matrix][i];
                //net->outputs[i] = sigmoid(tmp + net->biases[matrix][i]); //*****Temporary change for x^3 
                //net->outputs[i] = tmp + net->biases[matrix][i]; //sigmoid(tmp + net->biases[matrix][i]); //*****Temporary change for x^3
                tmp = 0;
            }
            //printf("wow");
        }
        else {
            for(int i = 0; i < net->hiddenSize; i++) {
                for(int j = 0; j < net->hiddenSize; j++) {
                    tmp = tmp + net->weights[matrix][j][i]*net->hidden[matrix-1][j];
                }
                if(net->sigmoid == 1) {
                    net->hidden[matrix][i] = sigmoid(tmp + net->biases[matrix][i]);
                }
                else if(net->relu == 1) {
                    net->hidden[matrix][i] = relu(tmp + net->biases[matrix][i]);
                }
                else {
                    net->hidden[matrix][i] = tmp + net->biases[matrix][i];
                }
                net->hiddenZ[matrix][i] = tmp + net->biases[matrix][i];
                //net->hidden[matrix][i] = sigmoid(tmp + net->biases[matrix][i]);
                tmp = 0;
            }
        } 
    }
}

Net createNetwork(int inputNeurons, int outputNeurons, int hiddenLayers, int hiddenSize, char* sigmoidOrRelu) {
    Net net;
    net.hiddenNum = hiddenLayers;
    net.hiddenSize = hiddenSize;
    net.inputNum = inputNeurons; //Specifies the number of input and output neurons
    net.outputNum = outputNeurons;
    if(strcmp(sigmoidOrRelu, "sigmoid") == 0) {
        net.sigmoid = 1;
        net.relu = 0;
    }
    else if(strcmp(sigmoidOrRelu, "relu") == 0) {
        net.sigmoid = 0;
        net.relu = 1;
    }
    else if(strcmp(sigmoidOrRelu, "none") == 0) {
        net.sigmoid = 0;
        net.relu = 0;
    }
    else {
        printf("Invalid non-linear function type");
        exit(1);
    }
    double variance = 2;
    net.inputs = (double*)malloc(inputNeurons*sizeof(double)); //Allocates memory for the input layer
    net.hidden = (double**)malloc(hiddenLayers*sizeof(double*));
    net.hiddenZ = (double**)malloc(hiddenLayers*sizeof(double*));
    for(int j = 0; j < hiddenLayers; j++) {
        net.hidden[j] = (double*)malloc(hiddenSize*sizeof(double));
        net.hiddenZ[j] = (double*)malloc(hiddenSize*sizeof(double));
    }
    net.outputs = (double*)malloc(outputNeurons*sizeof(double)); //allocates memory for output layer
    net.outputsZ = (double*)malloc(outputNeurons*sizeof(double));
    net.weights = createWeights(&net);
    net.biases = createBias(&net);    
    for(int matrix = 0; matrix<hiddenLayers+1; matrix++) { //Assigns random values to the weights
        if(hiddenLayers == 0) {
            for(int j = 0; j < inputNeurons; j++) {
                for(int i = 0; i < outputNeurons; i++) {
                    //net.weights[matrix][j][i] = ((double) rand() / RAND_MAX) * 2 - 1;
                    net.weights[matrix][j][i] = ((double) rand() / RAND_MAX) - 0.5;
                }
            }
        }
        else if(matrix == 0) {
            for(int j = 0; j < inputNeurons; j++) {
                for(int i = 0; i < hiddenSize; i++) {
                    net.weights[matrix][j][i] = ((double) rand() / RAND_MAX) - 0.5;
                }
            }
        }
        else if(matrix == hiddenLayers) {
            for(int j = 0; j < hiddenSize; j++) {
                for(int i = 0; i < outputNeurons; i++) {
                    net.weights[matrix][j][i] = ((double) rand() / RAND_MAX) - 0.5;
                }
            }
        }
        else {
            for(int j = 0; j < hiddenSize; j++) {
                for(int i = 0; i < hiddenSize; i++) {
                    net.weights[matrix][j][i] = ((double) rand() / RAND_MAX) - 0.5;
                }
            }
        }
    }
    for(int j = 0; j<hiddenLayers+1; j++) { //Assigns random values to the biases
        if(hiddenLayers == 0) {
            for(int i = 0; i<outputNeurons; i++) {
                //net.biases[j][i] = ((double) rand() / RAND_MAX) * 2 - 1;
                net.biases[j][i] = ((double) rand() / RAND_MAX) - 0.5;
            }
        }
        else if(j == 0) {
            for(int i = 0; i<hiddenSize; i++) {
                net.biases[j][i] = ((double) rand() / RAND_MAX) - 0.5;
            }
        }
        else if(j == hiddenLayers) {
            for(int i = 0; i<outputNeurons; i++) {
                net.biases[j][i] = ((double) rand() / RAND_MAX) - 0.5;
            }
        }
        else {
            for(int i = 0; i<hiddenSize; i++) {
                net.biases[j][i] = ((double) rand() / RAND_MAX) - 0.5;
            }
        }
    }
    return net;
}

void backProp(Net* net, Data* trainData, int batches, int batchSize, double learnRate){
    double* desired;
    int choice;
    double ***weights = createWeights(net);
    double **biases = createBias(net);
    double **actDeriv = createAct(net);
    int epoch = 0;
    int steps = 0;
    int count = 0;
    while(epoch < batches) {
        choice = rand() % trainData->rowCount;
        desired = getDesired(trainData->data[choice]); //Grabs row
        getInput(trainData->data[choice], net->inputs);
        compute(net);
        for(int matrix = net->hiddenNum; matrix >= 0; matrix--) { //j represents the current layer L. i represents previous layer L - 1. Iterates backwords through network matrices
            if(net->hiddenNum == 0) { //Network with no hidden layers
                for(int j = 0; j < net->outputNum; j++) {
                    for(int i = 0; i < net->inputNum; i++) { //Assumes no activation function in output layer
                        weights[0][i][j] = weights[0][i][j] + (net->inputs[i] * 2 * (net->outputs[j] - desired[j]));
                    }
                    biases[0][j] = biases[0][j] + (2 * (net->outputs[j] - desired[j]));
                }
            }
            else if(matrix == net->hiddenNum) { //Output layer -> hidden layer
                for(int j = 0; j < net->outputNum; j++) {
                    for(int i = 0; i < net->hiddenSize; i++) { //Assumes no activation function in output layer*********
                        weights[matrix][i][j] = weights[matrix][i][j] + (net->hidden[matrix-1][i] * 2 * (net->outputs[j] - desired[j]) * derivative(net, net->outputsZ[j]));
                        actDeriv[matrix-1][i] = actDeriv[matrix-1][i] + (net->weights[matrix][i][j] * 2 * (net->outputs[j] - desired[j]) * derivative(net, net->outputsZ[j])); //****
                        count++;
                    }
                    biases[matrix][j] = biases[matrix][j] + (2 * (net->outputs[j] - desired[j]) * derivative(net, net->outputsZ[j]));
                }
            }
            else if(matrix == 0) { //Hidden layer -> input layer
                for(int j = 0; j < net->hiddenSize; j++) {
                    for(int i = 0; i < net->inputNum; i++) {
                        weights[matrix][i][j] = weights[matrix][i][j] + (net->inputs[i] * derivative(net, net->hiddenZ[matrix][j]) * actDeriv[matrix][j]); //****
                    }
                    biases[matrix][j] = biases[matrix][j] + (derivative(net, net->hiddenZ[matrix][j]) * actDeriv[matrix][j]); //****
                }
            }
            else { //Hidden layer -> hidden layer
                for(int j = 0; j < net->hiddenSize; j++) {
                    for(int i = 0; i < net->hiddenSize; i++) {
                        weights[matrix][i][j] = weights[matrix][i][j] + (net->hidden[matrix-1][i] * derivative(net, net->hiddenZ[matrix][j]) * actDeriv[matrix][j]); //****
                        actDeriv[matrix-1][i] = actDeriv[matrix-1][i] + (net->weights[matrix][i][j] * derivative(net, net->hiddenZ[matrix][j]) * actDeriv[matrix][j]); //****
                    }
                    biases[matrix][j] = biases[matrix][j] + (derivative(net, net->hiddenZ[matrix][j]) * actDeriv[matrix][j]); //****
                }
            }
        }
        for(int row = 0; row < net->hiddenNum; row++) {
            for(int col = 0; col < net->hiddenSize; col++) {
                actDeriv[row][col] = 0;
            }
        }
        if(steps % batchSize == 0) {
            for(int matrix = 0; matrix < net->hiddenNum + 1; matrix++) {
                if(net->hiddenNum == 0) {
                    for(int j = 0; j < net->outputNum; j++) {
                        for(int i = 0; i < net->inputNum; i++) {
                            net->weights[0][i][j] = net->weights[0][i][j] - weights[0][i][j]/batchSize * learnRate;
                            weights[0][i][j] = 0;
                        }
                        net->biases[0][j] = net->biases[0][j] - biases[0][j]/batchSize * learnRate;
                        biases[0][j] = 0;
                    }
                }
                else if(matrix == 0) {
                    for(int j = 0; j < net->hiddenSize; j++) {
                        for(int i = 0; i < net->inputNum; i++) {
                            net->weights[matrix][i][j] = net->weights[matrix][i][j] - weights[matrix][i][j]/batchSize * learnRate;
                            weights[matrix][i][j] = 0;
                        }
                        net->biases[matrix][j] = net->biases[matrix][j] - biases[matrix][j]/batchSize * learnRate;
                        biases[matrix][j] = 0;
                    }
                }
                else if(matrix == net->hiddenNum) {
                    for(int j = 0; j < net->outputNum; j++) {
                        for(int i = 0; i < net->hiddenSize; i++) {
                            net->weights[matrix][i][j] = net->weights[matrix][i][j] - weights[matrix][i][j]/batchSize * learnRate;
                            weights[matrix][i][j] = 0;
                        }
                        net->biases[matrix][j] = net->biases[matrix][j] - biases[matrix][j]/batchSize * learnRate;
                        biases[matrix][j] = 0;
                    }
                }
                else {
                    for(int j = 0; j < net->hiddenSize; j++) {
                        for(int i = 0; i < net->hiddenSize; i++) {
                            net->weights[matrix][i][j] = net->weights[matrix][i][j] - weights[matrix][i][j]/batchSize * learnRate;
                            weights[matrix][i][j] = 0;
                        }
                        net->biases[matrix][j] = net->biases[matrix][j] - biases[matrix][j]/batchSize * learnRate;
                        biases[matrix][j] = 0;
                    }
                }
            }
            epoch++;
        }
        free(desired);
        steps++;
    }
}

void testNetwork(Net* net, Data* testData) {
    //double* desired;
    int score = 0;
    for(int i = 0; i < testData->rowCount; i++) {
        getInput(testData->data[i], net->inputs);
        compute(net);
        score = score + goodOutput(testData->data[i], net->outputs);
    }
    printf("%d/%d classified correctly", score, testData->rowCount);
}