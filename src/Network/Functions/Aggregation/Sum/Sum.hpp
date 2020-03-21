/*
 * Sum.hpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Jean-François Erdelyi
 */

#ifndef SUM_SUM_HPP_
#define SUM_SUM_HPP_

#include "../Aggregation.hpp"

class Sum: public Aggregation {
public:
	Sum();
	virtual double compute(const std::vector<std::shared_ptr<Link>>& inputLinks) override;
};

#endif /* SUM_SUM_HPP_ */
