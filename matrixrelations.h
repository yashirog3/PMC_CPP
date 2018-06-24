#ifndef MATRIXRELATIONS_H_
#define MATRIXRELATIONS_H_
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>

typedef std::vector<std::vector<double> > layer ;

/* ********************** RandomLayer Method: *********************************

Returns a matrix of lines by collumns with random double values between 0 and 1
Sintax: layer rndlayer = RandomLayer(linex, collumns);

********************************************************************************/

layer RandomLayer(int lines, int cols)
{
    layer ret;
    srand(time(nullptr));
    for(int i = 0; i < lines; i++)
    {
        std::vector<double> aux;
        for(int j = 0; j < cols; j++)
        {
            aux.push_back(rand() / (double) RAND_MAX);    
        }
        ret.push_back(aux);
    }
    return ret;
}

/************************ Transpose Method: ***********************************

Return a Matrix input Transposed
Sintax: layer TransposedLayer = Transpose(layer M);

********************************************************************************/

layer Transpose(layer W)
{
    layer ret;
    for(int j = 0; j < W[0].size(); j++)
    {
        std::vector<double> aux;
        for(int i = 0; i < W.size(); i++)
        {
            aux.push_back(W[i][j]);
        }        
        ret.push_back(aux);
    }
    return ret;
}

/************************** Matrix Method: *************************************

Perform a dot of two matrixes and return the result
Sintax: layer Dot = Matrix(layer L1, layer L2);

********************************************************************************/

std::vector<double> Matrix(std::vector<double> X, layer W)
{
    double Soma;
    if(X.size() == W.size())
    {
        layer WT = Transpose(W);
        std::vector<double> aux;
        for(int j = 0; j < WT.size(); j++)
        {
            Soma = 0;
            for(int i = 0; i < X.size(); i++)
            {
                Soma += X[i] * WT[j][i];
            }            
            aux.push_back(Soma);
        }
        return aux;
    }
}

/************************** LayerSigma Method: *********************************

Return an Array with the Output layer Distance of the desired value
Sintax: std::vector<double> OutputSigma = LayerSigma(std::vector<double> Desired, std::vector<double> Output);

********************************************************************************/

std::vector<double> LayerSigma(std::vector<double> Y, std::vector<double> Yi)
{
    std::vector<double> ret;
    for(int i = 0; i < Yi.size(); i++)
    {
        ret.push_back((Y[i] - Yi[i]) * (Yi[i] * (1.0 - Yi[i])));
    }
    return ret;
}

/*************************** LayerSigma Method: ************************************

Return an Array with the Not Output layer Distance of the desired value
Sintax: std::vector<double> NotOutputSigma = LayerSigma(std::vector<double> OutputError, layer WidthLayer,std::vector<double> InterOutput);

********************************************************************************/

std::vector<double> LayerSigma(std::vector<double> Sigma, layer W, std::vector<double> Y)
{
    std::vector<double> ret;
    double Soma;
    for(int j = 0; j < Y.size(); j++)
    {
        Soma = 0.0;
        for(int i = 0; i < W[j].size(); i++)
        {
            Soma += Sigma[j] * W[i][j];
        }
        ret.push_back(Soma * (Y[j] * (1.0 - Y[j])));
    }
    return ret;
}

/************************************************************** Update Method: ****************************************************************************

Update Weights Values with a delta rule
Sintax: Output(layer &Weights, std::vector<double> Error, std::vector<double> Inputs, double LearningRate);

************************************************************************************************************************************************************/
void Update(layer * W, std::vector<double> Sigma, std::vector<double> X, double tax=0.2)
{    
    for(int j = 0; j < W->size(); j++)
    {
        for(int i = 0; i < (*W)[j].size(); i++)
        {
            (*W)[j][i] += tax * Sigma[i] * X[j];
        }
    }
}

/******************************************************** Sigmoid Method: ****************************************************************************

Return a vector with a sigmoid function of an output
Sintax: std::vector<double> Output = Sigmoid(std::vector<double> Y)

*****************************************************************************************************************************************************/

std::vector<double> Sigmoid(std::vector<double> Y)
{
    std::vector<double> ret;
    for(int i = 0; i < Y.size(); i++)
    {
        ret.push_back(1.0 / (1.0 + exp(-Y[i])));
    }
    return ret;
}

/************************** CalcError Method: *********************************

Return the distance between an output and a desired value
Sintax: double Error = CalcError(std::vector<double> Desired, std::vector<double> Output)

********************************************************************************/
double CalcError(std::vector<double> Y, std::vector<double> Yi)
{ 
    double Error = 0;   
    for(int e = 0; e < Y.size(); e++)
        Error += 1./2 * pow((Y[e] - Yi[e]), 2);
}

#endif
