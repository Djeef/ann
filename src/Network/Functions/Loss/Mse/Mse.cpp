/*
 * Mse.cpp
 *
 *  Created on: Oct 26, 2019
 *      Author: Jean-François Erdelyi
 */

#include "Mse.hpp"

double Mse::compute(const std::vector<double>& outputErrors) {
	double sumSe = 0.0;
	size_t outputCount = outputErrors.size();

	// Compute the sum
	for (size_t i = 0; i < outputCount; ++i) {
		double error = outputErrors.at(i);
		sumSe += error * error;
	}

	// (1 / n) * sum(1..n)
	return (1 / (double) outputCount) * sumSe;
}
