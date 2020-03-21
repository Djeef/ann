/*
 * TanHyp.cpp
 *
 *  Created on: May 11, 2018
 *      Author: Jean-François Erdelyi
 */

#include <math.h>

#include "TanHyp.hpp"

TanHyp::TanHyp(): Excitation() {}

double TanHyp::compute(double x) {
	double epx = exp(x);
	double emx = exp(-x);
	return (epx - emx) / (epx + emx);
}

double TanHyp::derivative(double x) {
	double fx = compute(x);
	return 1 - (fx * fx);
}
