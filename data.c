#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "network.h"
#include <math.h>
#include <time.h>
#include "data.h"


double* getDesired(double* row) { //To be implemented depending on data set
    double* desired = (double*)malloc(10 * sizeof(double));
    for(int i = 0; i < 10; i++) {
        if(row[0] == i) {
            desired[i] = 1;
        }
        else {
            desired[i] = 0;
        }
    } 
    return desired;
}

void getInput(double* row, double* input) { //To be implemented depending on data set
    for(int i = 1; i < 785; i++) {
        input[i-1] = row[i];
    }
}

int goodOutput(double* row, double* output) { //To be implemented depending on data set
    int max = 0;
    for(int i = 0; i < 10; i++) {
        if(output[i] > output[max]) {
            max = i;
        }
    }
    if(max == row[0]) return 1;
    else return 0;
}

Data readCSV(char* fileName) { //Assumes Windows' standard of new line characters and that there's no new line at the last row
    FILE* fp = fopen(fileName, "r");
    if(fp == NULL) {
        printf("Invalid file");
        exit(1);
    }
    Data newData;
    char ch;
    newData.data = NULL;
    newData.columnCount = 0;
    newData.rowCount = 0;
    int count = 0;
    ch = fgetc(fp);
    while(ch != '\n') { //skips column titles
        ch = fgetc(fp);
        if(ch == ',') newData.columnCount++; //Counts the number of columns in data set
    }
    newData.columnCount++; //Preceding loop counts commas. There are commas + 1 columns in a dataset. 
    while(ch != EOF) { //Counts rows
        ch = fgetc(fp);
        if(ch == '\n') newData.rowCount++;
    }
    newData.rowCount++; //First row that's not counted
    fseek(fp, 0, SEEK_SET); //Goes back to start of file
    ch = fgetc(fp);
    while(ch != '\n') { //skips column titles
        ch = fgetc(fp);
    }
    newData.data = (double**)malloc(newData.rowCount * sizeof(double*)); //Alocates memory for the rows
    while(ch != EOF) {
        newData.data[count] = (double*)malloc(newData.columnCount * sizeof(double)); //Allocates a row
        for(int i = 0; i < newData.columnCount; i++) {
            fscanf(fp, "%lf", &newData.data[count][i]); //Loads in data
            ch = fgetc(fp); //Skips commas/newline
        }
        count++;
    }
    fclose(fp);
    return newData;
}


int main() {
    srand(time(NULL));

    Data trainData = readCSV("mnist_train.csv");
    Data testData = readCSV("mnist_test.csv");
    
    //Net new = createNetwork(784, 10, 1, 392, "sigmoid");
    Net new = open("mnistN.bin");
    backProp(&new, &trainData, 500, 10, 0.5);
    testNetwork(&new, &testData);
    //save(&new, "mnistN.bin");
    

    // getInput(testData.data[173], net.inputs);
    // compute(&net);
    // printf("[");
    // for(int i = 0; i < 10; i++) {
    //     printf("%f,", net.outputs[i]);
    // }
    // printf("]\n%f", testData.data[173][0]);
    return 1;
}