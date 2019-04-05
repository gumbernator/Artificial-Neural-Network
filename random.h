/*
	simple random number generator
	useful because neural net's weights initialize as a random numbers
	... and if you want you can do some NEAT(Neuroevolution of Augmented Topologies)
	... which is essentially a neural network with a genetic algorithm as an optimizer
*/

#include "include.h"

// returns random float between 'l' -> 'r'
float random(float l, float r) {
	if (l >= r) {
		cout<<"\nRANDOM LEFT > RIGHT !!! \n";
		return 0.0;
	}
	
	float gap = r - l;
	
	// generating random float ( 0.0 -> 1.0 )
	float rndm = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	
	rndm *= gap;
	
	return l + rndm;
}