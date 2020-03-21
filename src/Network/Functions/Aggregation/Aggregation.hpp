/*
 * Aggregation.hpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Jean-François Erdelyi
 */

#ifndef AGGREGATION_AGGREGATION_HPP_
#define AGGREGATION_AGGREGATION_HPP_

#include <vector>

#include "../../Link/Link.hpp"

class Link;

class Aggregation {
public:
	virtual double compute(const std::vector<std::shared_ptr<Link>>& inputLinks) = 0;
};

#endif /* AGGREGATION_AGGREGATION_HPP_ */
