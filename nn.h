#include "matrix.h"

using namespace std;

class NN 
{
private:
	// Main body of the Neural Net
	vector<Matrix> weight;
	vector<Matrix> layer;
	vector<Matrix> bias;
	
	// Structure of the Neural Net (Number of layers and their neurons)
	vector<int> structure;
  
	// the learning rate
	float lr;

	// optimizers
	/*
		0 -> Gradient Descent
		1 -> Momentum
		2 -> AdaGrad
	*/
	int optimizer;
	float momentumTerm;

	// loss functions
	/*
		0 -> MSE (Mean Squared Error)
		1 -> MAE (Mean Absolute Error)
	*/
	int lossFunc;

    // Previous Deltas for Momentum optimizer
	vector<Matrix> prevDeltaForWeight;
	vector<Matrix> prevDeltaForBias;

    // Sum of all Historic Gradient's square 
    // for AdaGrad
    vector<Matrix> GtForWeight;
    vector<Matrix> GtForBias;

    // small number preventing from division by zero
    float eta = 0.0000001;

public:
	// Constructor
	NN () {
		this->lr = 0.0;
		this->optimizer = -1;
		this->lossFunc = -1;
		this->momentumTerm = 0.9; // the defualt momentum term
		cout<<"Created a neural net\n";	
	};
	
	// Adds a layer at the back with given neurons and activation
	void addLayer(int, string);
	// Adds a layer at the back with given neurons and NO activation
	void addLayer(int);

	// Set structure
	void setStructure(vector<int>);
	
	// Prints structure
	void printStructure();
	
	// Inserting an Input
	void inputFloat(vector<float>);
	
	// Letting the neural net feed forward
	void feedforward();
	
	// Prints output
	void printOutput();

    // Resets Historic gradient
    // for AdaGrad
    void resetAdaGrad();

	// Returns prediction error based on given data
	float meanSqrError(vector<float>, vector<float>);
	
	// Copies itself and returns the copy
	NN copy();
	
	// Mutate Neural Net at given rate
	void mutate(float);

	// Set learning rate
	void setLearningRate(float);
	
	// Set Momentum term
	void setMomentumTerm(float);

	// Set optimizer
	void setOptimizer(string);

	// Set Loss function
	void setLossFunc(string);
	
	// Trains on given Input and target
	void train(vector<float>, vector<float>);
	
	// Trains on given Inputs and targets
	void train(vector<float>[], vector<float>[], int);
};

void NN::addLayer(int n, string s) 
{
	// updating structure
	this->structure.push_back(n);
	
	// adding weight and bias as a matrices if there are more than 1 layer
    // plus adding previous 
	if (this->structure.size() > 1) {
		Matrix tempWeight(n, this->structure[this->structure.size() - 2]);
		tempWeight.randomize(-2, 2);
		this->weight.push_back(tempWeight);

		Matrix tempBias(n, 1);
		tempBias.randomize(-2, 2);
		this->bias.push_back(tempBias);

		Matrix tempPrevDeltaForWeight(n, this->structure[this->structure.size() - 2]);
		tempPrevDeltaForWeight.fillZero();
		this->prevDeltaForWeight.push_back(tempPrevDeltaForWeight);

		Matrix tempPrevDeltaForBias(n, 1);
		tempPrevDeltaForBias.fillZero();
		this->prevDeltaForBias.push_back(tempPrevDeltaForBias);

        Matrix tempGt(n, n);
        tempGt.fillZero();
        this->GtForWeight.push_back(tempGt);
        this->GtForBias.push_back(tempGt);
	}
	
	// adding layer as matrix with one column
	Matrix tempLayer(n, 1);
	tempLayer.fillZero();
	tempLayer.setActivFunc(s);
	this->layer.push_back(tempLayer);
	cout<<"Added a layer with "<<n<<" neurons. (activation function: "<<s<<")\n";
}


void NN::addLayer(int n) 
{
	// updating structure
	this->structure.push_back(n);
	
	// adding weight and bias as a matrices if there are more than 1 layer
	if (this->structure.size() > 1) {
		Matrix tempWeight(n, this->structure[this->structure.size() - 2]);
		tempWeight.randomize(-2, 2);
		this->weight.push_back(tempWeight);

		Matrix tempBias(n, 1);
		tempBias.randomize(-2, 2);
		this->bias.push_back(tempBias);

		Matrix tempPrevDeltaForWeight(n, this->structure[this->structure.size() - 2]);
		tempPrevDeltaForWeight.fillZero();
		this->prevDeltaForWeight.push_back(tempPrevDeltaForWeight);

		Matrix tempPrevDeltaForBias(n, 1);
		tempPrevDeltaForBias.fillZero();
		this->prevDeltaForBias.push_back(tempPrevDeltaForBias);

        Matrix tempGt(n, n);
        tempGt.fillZero();
        this->GtForWeight.push_back(tempGt);
        this->GtForBias.push_back(tempGt);
	}
	
	// adding layer as matrix with one column
	Matrix tempLayer(n, 1);
	tempLayer.fillZero();
	this->layer.push_back(tempLayer);
	cout<<"Added a layer with "<<n<<" neurons. (no activation function)\n";
}

void NN::setStructure(vector<int> structure)
{
	this->structure = structure;
}

void NN::printStructure()
{
	// print out Structure
	cout<<endl<<"Printing Structure of the Neural Net..."<<endl;
	for (int i=0; i<this->structure.size(); i++)
		cout<<this->structure[i]<<"	";
	cout<<endl;
}

void NN::inputFloat(vector<float> input)
{
	// assigning input to the firt layer
	if (this->structure.size() > 0 && input.size() == this->structure[0]) {
		for (int i=0; i<input.size(); i++) {
			this->layer[0].setValue(i, 0, input[i]);
		}
	} else {
		cout<<"ERROR: INPUT size is not same as the first layer or NN is empty!!!\n";
		return;
	}
}

void NN::feedforward()
{
	// letting the network propagate forward
	if (this->layer.size() > 1) {
		for (int i = 1; i < this->layer.size(); i++) {
			this->layer[i] = this->weight[i-1] * this->layer[i-1] + this->bias[i-1];
			this->layer[i] = this->layer[i].activate();
		}
	}
    else 
    {
		cout<<"Forward Propagation failed. Neural net must have more than 1 layer\n";
    }
}

void NN::printOutput() 
{
	// printing output layer
	if (this->structure.size() > 1) 
    {
        cout<<"Printing output:\n";
		this->layer[this->structure.size()-1].print();
    }
	else 
		cout<<"Neural Net must have more than 1 layer\n";
}

float NN::meanSqrError(vector<float> input , vector<float> target)
{
	// Returning the mean of ( (target - output)^2 )
	if (this->layer[0].getM() == input.size() && this->structure[this->structure.size() - 1] == target.size()) 
    {
        this->inputFloat(input);
		this->feedforward();
		float sum = 0;
		for (int i=0; i<this->structure[this->structure.size() - 1]; i++)
			sum += (float)pow (this->layer[this->structure.size() - 1].getValue(i, 0) - target[i], 2.0);
		return sum / (float)this->structure[this->structure.size() - 1];
	} 
    else 
    {
		cout<<"meanSqrError error : Given input, target is not same dimensional as the Neural net's input, output layer\n";
	}
}

NN NN::copy()
{
	// copying itself to the "net" and returning it
	NN net;
	net.setStructure(this->structure);
	net.lr = this->lr;
	net.optimizer = this->optimizer;
    net.layer.clear();
    net.weight.clear();
    net.bias.clear();

    for (int i=0; i<this->layer.size(); i++) 
        net.layer.push_back(this->layer[i]);

	for (int i=0; i<this->weight.size(); i++) 
    net.weight.push_back(this->weight[i]);

	for (int i=0; i<this->bias.size(); i++) 
    net.bias.push_back(this->bias[i]);

    return net;
}

// modifying weights and biases randomly based on the mutation "rate"
void NN::mutate(float rate)
{
	for (int i=0; i<this->weight.size()-1; i++) {
      for (int i=0; i<this->weight[i].getM(); i++) {
        for (int j=0; j<this->weight[i].getN(); j++) {
          if (random(0, 1) < rate) {
            if (this->weight[i].getValue(i, j) != 0)
              this->weight[i].setValue(i, j, (float)random(-2, 2)*this->weight[i].getValue(i, j));
            else
              this->weight[i].setValue(i, j, (float)random(-2, 2));
          }
        }
      }
    }
  for (int i=0; i<this->bias.size()-1; i++) {
      for (int i=0; i<this->bias[i].getM(); i++) {
        for (int j=0; j<this->bias[i].getN(); j++) {
          if (random(0, 1) < rate) {
            if (this->bias[i].getValue(i, j) != 0)
              this->bias[i].setValue(i, j, (float)random(-2, 2)*this->bias[i].getValue(i, j));
            else
              this->bias[i].setValue(i, j, (float)random(-2, 2));
          }
        }
      }
    }
}

// setting learning rate
void NN::setLearningRate(float lr)
{
	this->lr = lr;
	if (this->lr > 0)
		cout<<"Set Learning rate: "<<lr<<endl;
	else
	{
		cout<<"Learning Rate must be positive!\n";
		exit(EXIT_FAILURE);
	}
}

// setting momentum term
void NN::setMomentumTerm(float mt)
{
	this->momentumTerm = mt;
	if (mt > 0)
		cout<<"Set Momentum Term: "<<mt<<endl;
	else
	{
		cout<<"Momentum Term must be positive!\n";
		exit(EXIT_FAILURE);
	}
}

// set Optimizer
void NN::setOptimizer(string s)
{
	if (s == "GD") {
		this->optimizer = 0;
		cout<<"Set Optimizer: Gradient Descent\n";
	}
	else if (s == "Momentum" || s == "momentum") {
		this->optimizer = 1;
		cout<<"Set Optimizer: Momentum\n";
		cout<<"Momentum term: "<<this->momentumTerm<<endl;
	}
	else if (s == "AdaGrad" || s == "Adagrad" || s == "adagrad")
	{
		this->optimizer = 2;
		cout<<"Set Optimizer: AdaGrad\n";
	}
	else
	{
		cout<<"Invalid Optimizer!\n";
		cout<<"Options:\n";
		cout<<"'GD': Gradient Descent\n";
		cout<<"'Momentum': Momentum\n";
		cout<<"'AdaGrad': Adaptive Gradient\n";
		exit(EXIT_FAILURE);
	}
}

// set Loss Function
void NN::setLossFunc(string s)
{
	if (s == "MSE") {
		this->lossFunc = 0;
		cout<<"Set Loss Function: Mean Squared Error\n";
	}
	else if (s == "MAE") {
		this->lossFunc = 1;
		cout<<"Set Loss Function: Mean Absolute Error\n";
	}
	else
	{
		cout<<"Invalid Loss Function!\n";
		cout<<"Options:\n";
		cout<<"'MSE': Mean Square Error\n";
		cout<<"'MAE': Mean Absolute Error\n";
		exit(EXIT_FAILURE);
	}
	
}

// training the neural net for given "input" and "target"
void NN::train(vector<float> input, vector<float> target)
{
	// checking if 'optimizer', 'loss function' and 'learning rate' are valid
	if (this->lr <= 0.0)
	{
		cout<<"Learning Rate is not defined!\n";
		cout<<"Learning Rate must be bigger than 0.0\n";
		exit(EXIT_FAILURE);
	}
	else if (this->optimizer == -1)
	{
		cout<<"Optimizer is not defined!\n";
		cout<<"Options:\n";
		cout<<"'GD': Gradient Descent\n";
		cout<<"'Momentum': Momentum\n";
		cout<<"'AdaGrad': Adaptive Gradient\n";
		exit(EXIT_FAILURE);
	}
	else if (this->lossFunc == -1)
	{
		cout<<"Loss function is not defined!\n";
		cout<<"Options:\n";
		cout<<"'MSE': Mean Square Error\n";
		cout<<"'MAE': Mean Absolute Error\n";
		exit(EXIT_FAILURE);
	}


	// checking if 'input' and 'target' is matching 
	// ...this Neural Net's input and output layer
	if (this->layer[0].getM() == input.size() && this->structure[this->structure.size() - 1] == target.size()) 
    {

        // Letting it feed forward therefore updating the layers
        this->inputFloat(input);
        this->feedforward();
        
        // just saving some typing
        int weightSize = this->weight.size();
        int layerSize = this->layer.size();

        // finding the error (loss) for every layers in Neural Net
            
        // converting the target from vector(n) to matrix(n, 1)
		// so we can do matrix math
        Matrix targetMatrix(target.size(), 1);
        for (int i=0; i<target.size(); i++)
            targetMatrix.setValue(i, 0, target[i]);

        // initializing errors and finding the error matrix for the output layer
		// propagating backwards
        Matrix error[weightSize];
        error[weightSize-1].setDimensions(this->layer[layerSize - 1].getM(), 1);
        error[weightSize-1] = (this->layer[layerSize - 1] - targetMatrix);

        // finding the error matrices for hidden layer
        for (int i=weightSize-2; i >= 0; i--) {
            error[i].setDimensions(this->weight[i+1].getN(), error[i+1].getN());
            error[i] = !this->weight[i+1] * error[i+1];
        }

        // updating weights according to the optimizer and loss function
        for (int i=0; i<weightSize; i++) 
		{
            Matrix out = this->layer[i+1];
            Matrix delta;

			if (this->lossFunc == 0)
			{
				Matrix tempDelta = error[i] ^ out.getDerivative();
				delta.setDimensions(tempDelta.getM(), tempDelta.getN());
				delta = tempDelta;
			}
			else if (this->lossFunc == 1)
			{
				Matrix tempDelta = out.getDerivative(error[i]);
				delta.setDimensions(tempDelta.getM(), tempDelta.getN());
				delta = tempDelta;
			}

            if (this->optimizer == 0)
            {
                Matrix deltaForBias = delta * this->lr;
                this->bias[i] = this->bias[i] - deltaForBias;

                Matrix deltaForWeight = (delta * !this->layer[i]) * this->lr;
                this->weight[i] = this->weight[i] - deltaForWeight;
            }
            else if (this->optimizer == 1)
            {
                Matrix deltaForBias = this->prevDeltaForBias[i] * this->momentumTerm + delta * this->lr;
                this->bias[i] = this->bias[i] - deltaForBias;
				this->prevDeltaForBias[i] = deltaForBias;

                Matrix deltaForWeight = this->prevDeltaForWeight[i] * this->momentumTerm + (delta * !this->layer[i]) * this->lr;
                this->weight[i] = this->weight[i] - deltaForWeight;
				this->prevDeltaForWeight[i] = deltaForWeight;
            }
            else if (this->optimizer == 2)
            {
                this->GtForBias[i] = this->GtForBias[i] + delta.squareDiag();
                for (int j=0; j<this->GtForBias[i].getM(); j++)
                    this->GtForBias[i].setValue(j, j, this->lr / sqrtf(this->GtForBias[i].getValue(j,j) + this->eta));

                Matrix deltaForBias = this->GtForBias[i] * delta;
                this->bias[i] = this->bias[i] - deltaForBias;

                this->GtForWeight[i] = this->GtForWeight[i] + (delta * !this->layer[i]).squareDiag();
                for (int j=0; j<this->GtForWeight[i].getM(); j++)
                    this->GtForWeight[i].setValue(j, j, this->lr / sqrtf(this->GtForWeight[i].getValue(j,j) + this->eta));

                Matrix deltaForWeight = this->GtForWeight[i] * (delta * !this->layer[i]);
                this->weight[i] = this->weight[i] - deltaForWeight;
            }
        }
	}
    else 
    {
		cout<<"Training error : Given input, target is not same dimensional as the Neural net's input, output layer\n";
	}
}

void NN::resetAdaGrad()
{
    for (int i=0; i<this->GtForBias.size(); i++)
    {
        this->GtForBias[i].fillZero();
        this->GtForWeight[i].fillZero();
    }
}