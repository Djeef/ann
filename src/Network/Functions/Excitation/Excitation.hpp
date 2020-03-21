/*
 * Excitation.hpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Jean-François Erdelyi
 */

#ifndef EXCITATION_EXCITATION_HPP_
#define EXCITATION_EXCITATION_HPP_

class Excitation {
public:
	virtual double compute(double x) = 0;
	virtual double derivative(double x) = 0;
};

#endif /* EXCITATION_EXCITATION_HPP_ */
