/*
 * Step.cpp
 *
 *  Created on: May 11, 2018
 *      Author: Jean-François Erdelyi
 */

#include "Step.hpp"

Step::Step(): Excitation() {}

double Step::compute(double x) {
	return (x < 0.0 ? 0.0 : 1.0);
}

double Step::derivative(double x) {
	return (x != 0.0 ? 0.0 : 1.0);
}

