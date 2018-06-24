#include "matrixrelations.h"

typedef std::vector<std::vector<double> > layer;

using namespace std;

int main(int argc, char ** argv)
{
	layer X = {{-1.0, 0.2, 0.9, 0.4},{-1.0, 0.1, 0.3, 0.5}, {-1.0, 0.9, 0.7, 0.8}, {-1.0, 0.6, 0.4, 0.3}};
	layer Y = {{0.7, 0.3},{0.6, 0.4},{0.9, 0.2},{0.2, 0.8}};
	layer W1 = RandomLayer(4,1000);
	layer W2 = RandomLayer(1001,10); 
	layer W3 = RandomLayer(11,2);
	std::vector<double> Y1, Y2, Y3, Sigma1, Sigma2, Sigma3;
	double Error = 0.0, Error1 = 1.0;
	int Epoch = 0;

	while(abs(Error - Error1) > 10e-10)   
	{    
	    Error1 = Error;
	    Error = 0.0;	    
	    for(int i = 0; i < X.size(); i++)
	    {
            //FeedForward pass
        	Y1 = Sigmoid(Matrix(X[i], W1));
	        Y1.insert(Y1.begin(), -1.0);
	        Y2 = Sigmoid(Matrix(Y1, W2));  
	        Y2.insert(Y2.begin(), -1.0);
        	Y3 = Sigmoid(Matrix(Y2, W3));

            //Error Calc
	        Error = CalcError(Y3, Y[i]);

            //Backpropagation pass
        	Sigma3 = LayerSigma(Y[i], Y3);        
	        Update(&W3, Sigma3, Y2, 0.5);       
        	Sigma2 = LayerSigma(Sigma3, W3, Y2);
	        Update(&W2, Sigma2, Y1, 0.5);
        	Sigma1 = LayerSigma(Sigma2, W2, Y1);
	        Update(&W1, Sigma1, X[i], 0.5);
	    }

	    Error /= Y.size();
	    Epoch++;
	}
	for(int j = 0; j < Y3.size(); j++)
	{
	    std::cout << Y3[j] << " ";
	}   
	return 0;
}

