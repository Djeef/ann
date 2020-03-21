/*
 * Sum.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Jean-François Erdelyi
 */

#include "Sum.hpp"

Sum::Sum(): Aggregation() {}

double Sum::compute(const std::vector<std::shared_ptr<Link>>& inputLinks) {
	double sum = 0.0;
	for (size_t i = 0; i < inputLinks.size(); ++i) {
		sum += inputLinks.at(i)->getComputedValue();
	}
	return sum;
}

