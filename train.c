#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "network.h"
// #define learnRate 0.0001


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

double reluDerivative(double x) {
    if(x > 0) return 1;
    return 0;
}

double sigDerivative(double x) {
    return exp(x)/((exp(x) + 1) * (exp(x) + 1));
}

double inverse(Net *net, double x) {
    if(net->relu == 1) { //****
        return x;
    }
    if(net->sigmoid == 1) {
        return -log(1/x - 1);
    }
    return x; //Linear activation
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

double function(double x) {
    return log(x);
    //return x * x + 2;
}

// double function(double x, double y, double z, double w) {
//     return 2*x + 3*y - 12*z - 2.4*w - 25;
// }



// double cost(double x) {
//     return (x-function(x))*(x-function(x));
// }

void backProp(Net *net, int batches, int batchSize, double learnRate){ //works with no hidden layers and one output neuron
    double desired[net->outputNum];
    double val1;
    double val2;
    double val3;
    double val4;
    double z; //The input into a neuron's activation function
    double ***weights = createWeights(net);
    double **biases = createBias(net);
    double **actDeriv = createAct(net);
    int epoch = 1;
    int steps = 0;
    while(epoch < batches) {
        val1 = rand() % 4 + 1; 
        //val1 = (rand() / RAND_MAX) * 5;
        //val1 = ((double) rand() / RAND_MAX) * 10 - 5;
        //val1 = rand() % 10;
        // val2 = rand() % 10;
        // val3 = rand() % 10;
        // val4 = rand() % 10;
        //desired = function(val);
        //desired[0] = function(val1, val2, val3, val4);
        desired[0] = function(val1);
        net->inputs[0] = val1;
        // net->inputs[1] = val2;
        // net->inputs[2] = val3;
        // net->inputs[3] = val4;
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
                    for(int i = 0; i < net->hiddenSize; i++) { //Assumes no activation function in output layer
                        weights[matrix][i][j] = weights[matrix][i][j] + (net->hidden[matrix-1][i] * 2 * (net->outputs[j] - desired[j]));
                        actDeriv[matrix-1][i] = actDeriv[matrix-1][i] + (net->weights[matrix][i][j] * 2 * (net->outputs[j] - desired[j])); //****
                    }
                    biases[matrix][j] = biases[matrix][j] + (2 * (net->outputs[j] - desired[j]));
                }
            }
            else if(matrix == 0) { //Hidden layer -> input layer
                for(int j = 0; j < net->hiddenSize; j++) {
                    z = inverse(net, net->hidden[matrix][j]); //Gets the activation function's input to use ****
                    for(int i = 0; i < net->inputNum; i++) {
                        weights[matrix][i][j] = weights[matrix][i][j] + (net->inputs[i] * derivative(net, z) * actDeriv[matrix][j]); //****
                    }
                    biases[matrix][j] = biases[matrix][j] + (derivative(net, z) * actDeriv[matrix][j]); //****
                }
            }
            else { //Hidden layer -> hidden layer
                for(int j = 0; j < net->hiddenSize; j++) {
                    z = inverse(net, net->hidden[matrix][j]); //****
                    for(int i = 0; i < net->hiddenSize; i++) {
                        weights[matrix][i][j] = weights[matrix][i][j] + (net->hidden[matrix-1][i] * derivative(net, z) * actDeriv[matrix][j]); //****
                        actDeriv[matrix-1][i] = actDeriv[matrix-1][i] + (net->weights[matrix][i][j] * derivative(net, z) * actDeriv[matrix][j]); //****
                    }
                    biases[matrix][j] = biases[matrix][j] + (derivative(net, z) * actDeriv[matrix][j]); //****
                }
            }
        }
        //memset(actDeriv, 0, sizeof(actDeriv));
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
        steps++;
    }
}

int main() {
    srand(time(NULL));
    Net test = createNetwork(1, 1, 2, 50, "relu");
    // for(int i = 0; i < 100; i++) {
    //     double val = ((double) rand() / RAND_MAX) * 4 - 2;
    //     printf("%f\n", val);
    // }
   //return 1;
    // for(int i = 0; i < 10; i++) {
    //     test.inputs[i] = 1;
    // }
    // save(&test, "test.bin");
    // Net test2 = open("test.bin");
    // for(int i = 0; i < 10; i++) {
    //     test2.inputs[i] = 1;
    // }
    // compute(&test);
    // compute(&test2);
    // printf("test: %f\ntest2: %f", test.outputs[0], test2.outputs[0]);
    // return 1;
    backProp(&test, 1000000, 15, 0.0001);
    // test.weights[0][0][0] = 2;
    // test.weights[0][1][0] = 3;
    // test.biases[0][0] = 1;
    test.inputs[0] = 3; 
    // test.inputs[1] = 3;
    // test.inputs[2] = 6;
    // test.inputs[3] = 1;
    compute(&test);
    //printf("Expected: %f\nActual: %f\n", function(13, 3, 6, 1), test.outputs[0]);
    printf("Expected: %f\nActual: %f\n", function(3), test.outputs[0]);
    //printf("Weight 1: %f\nWeight 2: %f\nBias 1: %f\nBias 2: %f\nbias 3: %f", test.weights[0][0][0], test.weights[1][0][0], test.biases[0][0], test.biases[1][0], test.biases[2][0]);
    save(&test, "test.bin");
    return 1;
}
