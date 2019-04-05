#include "nn.h"

int main() 
{
	
	// DO NOT ERASE THIS LINE !!!!!!!!
	srand (time(NULL));

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



	NN net;
	net.addLayer(2);
	net.addLayer(3, "sigmoid");
	net.addLayer(1, "sigmoid");
	string s;
	cin>>s;
	net.setOptimizer(s);
	net.setLearningRate(0.1);
	cin>>s;
	net.setLossFunc(s);

	for (int i=0; i<4; i++) {
        	   cout<<data[i][0]<<"  "<<data[i][1]<<endl;
        	   net.inputFloat(data[i]);
        	   net.feedforward();
        	   net.printOutput();
	}

	// vector<Matrix> temp;
	// Matrix t;
	// t.setDimensions(2,2);
	// temp.push_back(t);
	// t.setDimensions(3,3);
	// temp.push_back(t);

	// temp[0].print();
	// temp[1].print();

	float error = 99999999;
	int counter = 0;
	float lr = 0.1;
    while (error > 0.005) {
        int select = (int)random(0, 4);
        net.train(data[select], target[select]);
            
		float sum = 0;
		for (int i=0; i<4; i++)
			sum += net.meanSqrError(data[i], target[i]);
		error = sum / 4.0;
		cout<<"error: "<<error*100<<"%"<<"\r";

		counter++;
		if (counter % 40 == 0)
		{
			net.resetAdaGrad();
		}
    }
	cout<<"\n";

	for (int i=0; i<4; i++) {
	   cout<<data[i][0]<<"  "<<data[i][1]<<endl;
	   net.inputFloat(data[i]);
	   net.feedforward();
	   net.printOutput();
	}

	return 0;
}