/*
 * Sigmoid.hpp
 *
 *  Created on: Mar 9, 2018
 *      Author: Jean-François Erdelyi
 */

#ifndef SIGMOID_SIGMOID_HPP_
#define SIGMOID_SIGMOID_HPP_

#include "../Excitation.hpp"

class Sigmoid: public Excitation {
protected:
	double lambda;

public:
	Sigmoid(double lambda = 1.0);
	virtual double compute(double x) override;
	virtual double derivative(double x) override;
};

#endif /* SIGMOID_SIGMOID_HPP_ */
