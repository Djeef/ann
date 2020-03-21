/*
 * Sigmoid.cpp
 *
 *  Created on: Mar 9, 2018
 *      Author: Jean-François Erdelyi
 */

#include <math.h>

#include "Sigmoid.hpp"

Sigmoid::Sigmoid(double lambda): Excitation() {
	this->lambda = lambda;
}

double Sigmoid::compute(double x) {
	return 1 / (1 + exp(-(lambda * x)));
}

double Sigmoid::derivative(double x) {
	double y = compute(x);
	return y * (1 - y);
}
