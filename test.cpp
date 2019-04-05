#include "nn.h"

int main() 
{
	
	// DO NOT ERASE THIS LINE !!!!!!!!
	srand (time(NULL));


	// preparing data (XOR)
	// input -> output
	// 0 0 -> 0
	// 0 1 -> 1
	// 1 0 -> 1
	// 1 1 -> 0
	vector<float> data[4];
	data[0].push_back(0);
	data[0].push_back(0);

	data[1].push_back(0);
	data[1].push_back(1);

	data[2].push_back(1);
	data[2].push_back(0);

	data[3].push_back(1);
	data[3].push_back(1);
	
	vector<float> target[4];
	target[0].push_back(0);

	target[1].push_back(1);

	target[2].push_back(1);

	target[3].push_back(0);


	// initializing Neural Net
	NN net;

	// adding first layer (input layer) with 2 neurons
	net.addLayer(2);
	
	// adding a layer (first hidden layer) with 3 neurons
	// activation function of this layer is sigmoid
	net.addLayer(3, "sigmoid");
	
	// adding a layer (which can be another hidden layer if you wanna go deep)
	// activation if this layer is also 
	net.addLayer(1, "sigmoid");

	// you can add how many layers you like :)

	/* 
		setting the optimizer for backpropagation. options:
		"GD" (simple form)
		"Momentum" (using Momentum to update the parameters)
		"AdaGrad" (Decaying the learning rate using sum of squared Historic parameters) 

		in this case we used "AdaGrad"
	*/ 
	net.setOptimizer("AdaGrad");

	// setting the learning rate to 0.01
	net.setLearningRate(0.01);
	
	/*
		setting the loss function for out output. options:
		"MAE" (Mean Absolute Error)
		"MSE" (Mean Squared Error)
	 	
		in this case "MSE" (Mean Square Error)
	*/
	net.setLossFunc("MSE");


	// let's see how well our network is going to perform
	// it's gonna be so bad :P
	for (int i=0; i<4; i++) {
        	   cout<<data[i][0]<<"  "<<data[i][1]<<endl;
        	   net.inputFloat(data[i]);
        	   net.feedforward();
        	   net.printOutput();
	}

	float error = 99999999;
	int counter = 0;

	// now we train!!!
    while (error > 0.005) {
        int select = (int)random(0, 4);
		// training on a random samples of our xor dataset 
        net.train(data[select], target[select]);

		// calculating the mean square error of the output
		float sum = 0;
		for (int i=0; i<4; i++)
			sum += net.meanSqrError(data[i], target[i]);
		error = sum / 4.0;
		
		// printing the error
		cout<<"error: "<<error*100<<"%"<<"\r";

		/*
			Every 40 steps we need to reset the AdaGrad's historic sum
			...preventing our learning rate to become insignificantly small
			It's called Tuning learning rate
		*/
		counter++;
		if (counter % 40 == 0)
		{
			net.resetAdaGrad();
		}

		// we will train until the error becomes less than 0.5%
    }
	cout<<"\n";

	// Now we can see it is doing quite well
	for (int i=0; i<4; i++) {
	   cout<<data[i][0]<<"  "<<data[i][1]<<endl;
	   net.inputFloat(data[i]);
	   net.feedforward();
	   net.printOutput();
	}

	return 0;
}