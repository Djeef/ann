/*
 * TanHyp.hpp
 *
 *  Created on: May 11, 2018
 *      Author: Jean-François Erdelyi
 */

#ifndef SRC_NETWORK_FUNCTIONS_EXCITATION_TANHYP_TANHYP_HPP_
#define SRC_NETWORK_FUNCTIONS_EXCITATION_TANHYP_TANHYP_HPP_

#include "../Excitation.hpp"

class TanHyp: public Excitation {
public:
	TanHyp();
	virtual double compute(double x) override;
	virtual double derivative(double x) override;
};

#endif /* SRC_NETWORK_FUNCTIONS_EXCITATION_TANHYP_TANHYP_HPP_ */
