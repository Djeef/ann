/*
 * Step.hpp
 *
 *  Created on: May 11, 2018
 *      Author: Jean-François Erdelyi
 */

#ifndef SRC_NETWORK_FUNCTIONS_EXCITATION_STEP_STEP_HPP_
#define SRC_NETWORK_FUNCTIONS_EXCITATION_STEP_STEP_HPP_

#include "../Excitation.hpp"

class Step: public Excitation {
public:
	Step();
	virtual double compute(double x) override;
	virtual double derivative(double x) override;
};

#endif /* SRC_NETWORK_FUNCTIONS_EXCITATION_STEP_STEP_HPP_ */
