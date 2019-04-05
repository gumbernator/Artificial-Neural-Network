/*

	A simple matrix function that I think anyone can understand :)
	If my code looks fuzzy, please let me know at guyugmonkh@hotmail.com

*/

#include "random.h"

class Matrix
{
 private:
 	// row
 	int m;
 	
 	// column
 	int n;
 	
 	// values of a matrix
 	vector<vector<float>> values;

	// activation function indicator
	/*
		0 -> sigmoid
		1 -> tanh
	*/
	int activFunc;
 	
 public:
 	// constructor
 	Matrix() {
		this->activFunc = -1;
		this->m = 0;
		this->n = 0;
	}
 	
 	// constructor with parameters
 	Matrix(int, int);

	Matrix copy();
	
	// prints values
	void print();
	
	// returns width
	int getM() {return this->m;}
	
	// returns height
	int getN() {return this->n;}
	
	// returns a value at the given position
	float getValue(int a, int b) {return this->values[a][b];}

	// return activation function
	int getActivFunc() {return this->activFunc;}

	// return derivative
	Matrix getDerivative();

	// return derivative for MAE where elements are derived differently
	// ... based on error matrix
	Matrix getDerivative(Matrix);
	
	// sets width, height and declares values
	void setDimensions(int, int);
	
	// sets a value at the given position
	void setValue(int a, int b, float c) {this->values[a][b] = c;}
	
	// fill up the values with zero
	void fillZero();
	
	// randomize elements
	void randomize(float, float);

	// choose activation function
	void setActivFunc(string);

	// update values according to Activation function
	Matrix activate();

	Matrix squareDiag();
	
	// redifinition of the operators
	
	// element wise addition
	Matrix operator +(Matrix);
	Matrix operator +(float);
	
	// element wise substraction
	Matrix operator -(Matrix);
	Matrix operator -(float);
	
	// Matrix dot multiplication
	Matrix operator *(Matrix);
	Matrix operator *(float);
	
	// assignation
	void operator = (Matrix);
	
	// tranpose
	Matrix operator !();
	
	// element wise multiplication
	Matrix operator ^(Matrix);
};

Matrix::Matrix(int a, int b) 
{
		// allocating 2d array "values[a][b]" with random values
		
		this->values.resize(a);
		for (int i=0; i<a; i++)
			this->values[i].resize(b);

		// values = new float*[a];
		
		// for(int i=0; i<a; i++) {
		// 	for(int j=0; j<b; j++) {
		// 		values[i] = new float[j];
		// 	}
		// }
		
		// passing parameters onto a width and height
		this->m = a;
		this->n = b;

		this->activFunc = -1;
}

Matrix Matrix::copy()
{
	Matrix temp(this->m, this->n);
		for(int i=0; i<this->m; i++) {
			for(int j=0; j<this->n; j++) {
				float ex = exp(this->values[i][j]);
				float e_x = 1 / ex;

				temp.setValue(i,j, (ex - e_x) / (ex + e_x));
			}
		}
	return temp;
}

void Matrix::fillZero() 
{
	
		// filling it up with zero
		for(int i=0; i<this->m; i++) {
			for(int j=0; j<this->n; j++) {
				this->values[i][j] = 0.0;
			}
		}
}

void Matrix::randomize(float left, float right)
{
	// randomizing elements
	
		for(int i=0; i<this->m; i++) {
			for(int j=0; j<this->n; j++) {
				this->values[i][j] = random(left, right);
			}
		}
}

void Matrix::print() 
{
	for(int i=0; i<this->m; i++) {
		for(int j=0; j<this->n; j++) {
			cout<<"	"<<this->values[i][j];
		}
		 cout<<endl;
	}
	 cout<<endl;
}

void Matrix::setDimensions(int a, int b) 
{
	// resetting values
	this->m = a;
	this->n = b;
	this->values.resize(a);
	for (int i=0; i<a; i++)
		this->values[i].resize(b);
	// this->values = new float*[a];
	// for(int i=0; i<a; i++) {
	// 	for(int j=0; j<b; j++) {
	// 		this->values[i] = new float[j];
	// 	}
	// }
}

Matrix Matrix::operator +(Matrix a) 
{
		
		// checks if two matrixes have same width and height
		if (a.getM() == this->m && a.getN() == this->n) {
			
			// declares a product matrix
			Matrix product;
			
			// giving a product matrix's width and height
			product.setDimensions(this->m, this->n);
			
			// setting a product matrix's values
			for(int i=0; i<this->m; i++) {
 				for(int j=0; j<this->n; j++) {
 					product.setValue(i,j, (this->values[i][j] + a.getValue(i,j)));
			 	}
		 	}
			//returning a product
		 	return product;
		 	
		} else {
			cout<<"+ MATRIX ERROR"<<endl;
		}
	}
Matrix Matrix::operator +(float a) 
{
			
			// declares a product matrix
			Matrix product;
			
			// giving a product matrix's width and height
			product.setDimensions(this->m, this->n);
			
			// setting a product matrix's values
			for(int i=0; i<this->m; i++) {
 				for(int j=0; j<this->n; j++) {
 					product.setValue(i,j, (this->values[i][j] + a));
			 	}
		 	}
		 	
		 	return product;
		 	
}
Matrix Matrix::operator *(Matrix a) 
{
	
		// checks if two matrixes can be multiplied
		if (a.getM() == this->n) {
			
			// declares a product matrix
			Matrix product;
			
			// giving a product matrix's width and height
			product.setDimensions(this->m, a.getN());
			
			// setting a product matrix's values
			for(int i=0; i<this->m; i++) {
 				for(int j=0; j<a.getN(); j++) {
 					float c=0;
 					for(int k=0; k<this->n; k++) {
 						c += this->values[i][k] * a.getValue(k,j);
					 }
 					product.setValue(i,j, c);
			 	}
		 	}
			//returning a product
		 	return product;
		 	
		} else {
			cout<<"* MATRIX ERROR"<<endl;
		}
}
Matrix Matrix::operator *(float a)
{
		// declares a product matrix
			Matrix product;
			
			// giving a product matrix's width and height
			product.setDimensions(this->m, this->n);
			
			// setting a product matrix's values
			for(int i=0; i<this->m; i++) {
 				for(int j=0; j<this->n; j++) {
 					product.setValue(i,j, this->values[i][j] * a);
			 	}
		 	}
			
			return product;
}
Matrix Matrix::operator -(Matrix a) 
{
	
		// checks if two matrixes have same width and height
		if (a.getM() == this->m && a.getN() == this->n) {
			
			// declares a product matrix
			Matrix product;
			
			// giving a product matrix's width and height
			product.setDimensions(this->m, this->n);
			
			// setting a product matrix's values
			for(int i=0; i<this->m; i++) {
 				for(int j=0; j<this->n; j++) {
 					product.setValue(i,j, (this->values[i][j] - a.getValue(i,j)));
			 	}
		 	}
			//returning a product
		 	return product;
		 	
		} else {
			cout<<"- MATRIX ERROR"<<endl;
		}
}
Matrix Matrix::operator -(float a) 
{
			
			// declares a product matrix
			Matrix product;
			
			// giving a product matrix's width and height
			product.setDimensions(this->m, this->n);
			
			// setting a product matrix's values
			for(int i=0; i<this->m; i++) {
 				for(int j=0; j<this->n; j++) {
 					product.setValue(i,j, (this->values[i][j] - a));
			 	}
		 	}
		 	
		 	return product;
		 	
}
void Matrix::operator =(Matrix a) 
{
	// checks if two matrixes have same width and height
		if (a.getM() == this->m && a.getN() == this->n) {
			
			// setting a matrix's values
			for(int i=0; i<this->m; i++) {
 				for(int j=0; j<this->n; j++) {
 					this->setValue(i,j, (a.getValue(i,j)));
			 	}
		 	}
		 	
		} else {
			cout<<"= MATRIX ERROR"<<endl;
		}
}
Matrix Matrix::operator !() 
{
		// declares a product matrix
			Matrix product;
			
			// giving a product matrix's width and height
			product.setDimensions(this->n, this->m);
			
			// setting a product matrix's values
			for(int i=0; i<this->n; i++) {
 				for(int j=0; j<this->m; j++) {
 					product.setValue(i,j, (this->values[j][i]));
			 	}
		 	}

			 return product;
		 	// //  deleting previous values
		 	// for(int i; i<this->m; i++) {
		 	// 	delete [] this->values[i];
			// }
		 	// delete [] this-> values;
			 
			// // setting dimensions and adding new values
			//  this->setDimensions(product.getM(), product.getN());
			 
			//  for(int i=0; i<product.getM(); i++) {
 			// 	for(int j=0; j<product.getN(); j++) {
 			// 		this->values[i][j] = product.getValue(i, j);
			//  	}
		 	// }
}
Matrix Matrix::operator ^(Matrix a) 
{
		// checks if two matrixes have same width and height
		if (a.getM() == this->m && a.getN() == this->n) {
			
			// declares a product matrix
			Matrix product;
			
			// giving a product matrix's width and height
			product.setDimensions(this->m, this->n);
			
			// setting a product matrix's values
			for(int i=0; i<this->m; i++) {
 				for(int j=0; j<this->n; j++) {
 					product.setValue(i,j, (this->values[i][j] * a.getValue(i,j)));
			 	}
		 	}
			//returning a product
		 	return product;
		 	
		} else {
			cout<<"^(element wise multiplication) MATRIX ERROR"<<endl;
		}
}

void Matrix::setActivFunc(string s)
{
	if (s == "sigmoid")
		this->activFunc = 0;
	else if (s == "tanh")
		this->activFunc = 1;
	else {
		this->activFunc = -1;
		cout<<"fucked up!!!\n";
	}
}

Matrix Matrix::activate()
{
	if (this->activFunc == -1)
	{
		return this->copy();
	}
	else if (this->activFunc == 0) 
	{
		Matrix temp(this->m, this->n);
		for(int i=0; i<this->m; i++) {
			for(int j=0; j<this->n; j++) {
				temp.setValue(i,j, 1.0 / (1.0 + (float)exp(-this->values[i][j])));
			}
		}
		return temp;
	} 
	else if (this->activFunc == 1) 
	{
		Matrix temp(this->m, this->n);
		for(int i=0; i<this->m; i++) {
			for(int j=0; j<this->n; j++) {
				float ex = exp(this->values[i][j]);
				float e_x = 1 / ex;

				temp.setValue(i,j, (ex - e_x) / (ex + e_x));
			}
		}
		return temp;
	}
}

Matrix Matrix::getDerivative()
{
	if (this->activFunc == -1) 
	{
		Matrix temp(this->m, this->n);
		for(int i=0; i<this->m; i++) {
			for(int j=0; j<this->n; j++) {
				temp.setValue(i,j, 1.0);
			}
		}
		return temp;
	}
	else if (this->activFunc == 0)
	{
		Matrix temp = this->copy();
		temp = temp ^ (temp * (-1.0) + 1.0);
		return temp;
	}
	else if (this->activFunc == 1) 
	{
		Matrix temp = this->copy();
		temp = ((temp ^ temp) * (-1.0) + 1.0);
		return temp;
	}
}

Matrix Matrix::getDerivative(Matrix e)
{
	if (this->activFunc == -1) 
	{
		Matrix temp(this->m, this->n);
		for(int i=0; i<this->m; i++) {
			for(int j=0; j<this->n; j++) {
				if (e.getValue(i, 0) >= 0)
					temp.setValue(i,j, 1.0);
				else 
					temp.setValue(i,j, -1.0);
			}
		}
		return temp;
	}
	else if (this->activFunc == 0)
	{
		Matrix temp = this->copy();
		temp = temp ^ (temp * (-1.0) + 1.0);
		for(int i=0; i<this->m; i++) {
			for(int j=0; j<this->n; j++) {
				if (e.getValue(i, 0) >= 0)
					temp.setValue(i,j, temp.getValue(i,j));
				else 
					temp.setValue(i,j, -temp.getValue(i,j));
			}
		}
		return temp;
	}
	else if (this->activFunc == 1) 
	{
		Matrix temp = this->copy();
		temp = ((temp ^ temp) * (-1.0) + 1.0);
		for(int i=0; i<this->m; i++) {
			for(int j=0; j<this->n; j++) {
				if (e.getValue(i, 0) >= 0)
					temp.setValue(i,j, temp.getValue(i,j));
				else 
					temp.setValue(i,j, -temp.getValue(i,j));
			}
		}
		return temp;
	}
}

Matrix Matrix::squareDiag()
{
	Matrix temp(this->m, this->m);
	temp.fillZero();

	for (int i=0; i<this->m; i++)
		for (int j=0; j<this->n; j++)
			temp.setValue(i,i, temp.getValue(i,i) + powf(this->values[i][j], 2));

	return temp;
}