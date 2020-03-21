/*
 * Loss.hpp
 *
 *  Created on: Sep 22, 2019
 *      Author: Jean-François Erdelyi
 */

#ifndef LOSS_LOSS_HPP_
#define LOSS_LOSS_HPP_

#include <vector> 

class Loss {
public:
	virtual double compute(const std::vector<double>& outputErrors) = 0;
};

#endif /* LOSS_LOSS_HPP_ */
