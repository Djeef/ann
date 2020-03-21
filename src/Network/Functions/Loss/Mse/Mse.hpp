/*
 *	Mse.hpp
 *
 *  Created on: Oct 26, 2019
 *      Author: Jean-François Erdelyi
 */

#ifndef MSE_MSE_HPP_
#define MSE_MSE_HPP_

#include "../Loss.hpp"

class Mse : public Loss {
	double compute(const std::vector<double>& outputErrors) override;
};

#endif /* MSE_MSE_HPP_ */
