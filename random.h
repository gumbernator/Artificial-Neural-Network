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