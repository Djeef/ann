/*
 * Neural.hpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Jean-François Erdelyi
 */

#ifndef NEURAL_NEURAL_HPP_
#define NEURAL_NEURAL_HPP_

#include <vector>
#include <string>

#include "../../constant.h"

#include "../Functions/Aggregation/Aggregation.hpp"
#include "../Functions/Excitation/Excitation.hpp"

class Link;
class Aggregation;

/**
TODO: create abstract class and one classe by type: 
 - INPUT
 - OUTPUT
 - HIDDEN
 - BIAIS
 - <Maybe other type in the future>
*/

/**
 * Representation of one neural in the network 
 */
class Neural: public std::enable_shared_from_this<Neural> {
protected:
	/**
	 * Type
	 */
	NeuralType type;

	/**
	 * Bias neural
	 */
	std::shared_ptr<Neural> biasNeural;

	/**
	 * Neural bias link
	 */
	std::shared_ptr<Link> biasLink;

	/** 
	 * Aggragation function
	 */
	std::unique_ptr<Aggregation> aggregationFunction;

	/**
	 * Excitation function
	 */
	std::unique_ptr<Excitation> excitationFunction;

	/**
	 * Input links
	 */
	std::vector<std::shared_ptr<Link>> inputLinks;
	
	/**
	 * Output links
	 */
	std::vector<std::shared_ptr<Link>> outputLinks;

	/**
	 * Neural output value
	 */
	double outputValue;

	/**
	 * Neural input value
	 */
	double inputValue;

	/**
	 * Sum of delta from the next layer (delta * w)
	 */
	double nextLayerComputedDelta;

	/**
	 * Delta
	 */
	double delta;

protected:
	/**
	 * Set delta (sum(w * delta)[n+1] * f'(value))
	 */
	void setDelta();

	/**
	 * Compute weight
	 * @throw logic_error bias/input neural can not be computed
	 */
	void computeWeight();

public:

	/**
	 * Constructor
	 * @param t neural type
	 * @param nbN number of neurals
	 * @param aT aggregation type
	 * @param eT excitation type
	 * @param v default value
	 */
	Neural(NeuralType t, AggregationType aT, ExcitationType eT, double v = 0.0);

	/**
	 * Add bias
	 * @param value bias value (usualy 1.0 or -1.0)
	 */
	void addBias(double value);

	/**
     * Destructor
     */
	~Neural() = default;

	/**
	 * Add input link
	 * @param link input link
	 * @throw logic_error bias/input neural have no input links
	 */
	void addInputLink(std::shared_ptr<Link> link);

	/**
	 * Add output link
	 * @param link output link
	 * @throw logic_error output neural have no output links
	 */
	void addOutputLink(std::shared_ptr<Link> link);

	/**
	 * Connect to next neural
	 * @param next next neural
	 * @retrun the created output link
	 */
	std::shared_ptr<Link>& connectTo(std::shared_ptr<Neural>& next);

	/**
	 * Connect to previous neural
	 * @param next previous neural
	 * @retrun the created input link
	 */
	std::shared_ptr<Link>& connectFrom(std::shared_ptr<Neural>& previous);

	/**
	 * Compute forward propagation
	 * @throw logic_error bias/input neural can not be computed
	 */
	void forwardPropagation();

	/**
     * Compute backward propagation
	 * @throw logic_error bias/input neural can not be computed
	 */
	void backwardPropagation();

	/**
	 * Compute output delta
	 * @param expectedValue expected value
	 * @retun delta output
	 * @throw logic_error only output layer can compute delta output
	 */
	double computeDeltaOutput(double expectedValue);

	/**
	 * Set value of the neural (only input neural can be set)
	 * @param value value of the neural
	 * @throw logic_error if the neural is not an input neural
	 */
	void setValue(double value);

	/**
	 * Get delta error
	 * @return delta error
	 */
	double getDelta();

	/**
	 * Get neural value
	 * @return neural value
	 */
	double getOutputValue();

	/**
	 * Get bias neural
	 * @return bias neural
	 */
	std::shared_ptr<Neural>& getBiasNeural();

	/**
	 * Get bias link
	 * @return bias link
	 */
	std::shared_ptr<Link>& getBiasLink();

	/**
	 * To string
	 * @return neural in string
	 */
	std::string toString();
};

#endif /* SRC_NEURAL_NEURAL_HPP_ */
