/*
 * Link.hpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Jean-François Erdelyi
 */

#ifndef LINK_LINK_HPP_
#define LINK_LINK_HPP_

#include <iostream>
#include <string>

#include "../../constant.h"
#include "../Neural/Neural.hpp"

class Neural;

/**
 * Representation of link between two neurals
 */
class Link: public std::enable_shared_from_this<Link> {
private:
	/**
	 * Alpha value used in weight computation, usualy between 0.0 and 1.0 [default = 0.01]
	 */
	static double learningRate;

protected:
	/**
	 * Prervious neural
	 */
	std::shared_ptr<Neural> previousNeural;

	/**
	 * Next neural
	 */
	std::shared_ptr<Neural> nextNeural;

	/**
	 * Weight of the link
	 */
	double weight;

public:
	/**
	 * Constructor
	 * @param previousNeural previous neural
	 * @param nextNeural next neural
	 */
	Link(std::shared_ptr<Neural> previousNeural, std::shared_ptr<Neural> nextNeural);

	/**
	 * Connect to neural depending of the direction 
	 * @param direction can be FORWARD, BACKWARD or BOTH
	 */
	void connect(ConnectDirection direction);

	/**
	 * Get computed value value * weight
	 * @return computed value
	 */
	double getComputedValue();

	/**
	 * Get computed delta delta * weight
	 * @return computed delta
	 */
	double getComputedDelta();
	
	/**
	 * Get previous value
	 * @return previous value
	 */
	double getPreviousValue();
	
	/**
	 * Get next value
	 * @return next value
	 */
	double getNextDelta();

	/**
	 * Compute new weight
	 */
	void computeWeight();

	/**
	 * Set learning rate for all links
	 * @param learningRate alpha value used in weight computation
	 */
	static void setLearningRate(double learningRate);

	/**
	 * To string
	 * @return link in string
	 */
	std::string toString();
};

#endif /* LINK_LINK_HPP_ */
