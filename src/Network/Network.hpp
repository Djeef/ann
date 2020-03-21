/*
 * Network.hpp
 *
 *  Created on: Mar 6, 2018
 *      Author: Jean-François Erdelyi
 *
 * @link https://towardsdatascience.com/how-does-back-propagation-in-artificial-neural-networks-work-c7cad873ea7
 */

#ifndef NETWORK_NETWORK_HPP_
#define NETWORK_NETWORK_HPP_

#include <vector>

#include "../constant.h"

#include "Neural/Neural.hpp"
#include "Functions/Loss/Loss.hpp"

/**
 * The neural handler the list of all neurals
 */
class Network {
protected:
	/**
	 * Is input layer set
	 */
	bool isInputSet;

	/**
	 * Is output layer set
	 */
	bool isOutputSet;

	/**
	 * Neurals count
	 */
	unsigned int neuralsCount;

	/**
	 * Links count
	 */
	unsigned int linksCount;

	/*
	 * Number of output neurals
	 */
	unsigned int outputsCount;

	/**
	 * Last index of neurals layers
	 */
	unsigned int lastNeuralLayerIndex;

	/**
	 * The representation of neural layers
	 */
	std::vector<std::vector<std::shared_ptr<Neural>>> neurals;

	/**
	 * The representation of neural links
	 */
	std::vector<std::vector<std::shared_ptr<Link>>> links;

	/**
	 * Result of loss function
	 */
	double loss;

	/**
	 * Loss function
	 */
	std::unique_ptr<Loss> lossFunction;

protected:
	/**
	 * Compute back propagation
	 */
	void backPropagation();

	/**
	 * Get class max
	 * @return max class
	 */
	int getClassMax();

	/**
	 * Add layer
	 * @param t neural type
	 * @param nbN number of neurals
	 * @param aT aggregation type
	 * @param eT excitation type
	 */
	void addLayer(NeuralType t, unsigned int nbN, AggregationType aT, ExcitationType eT);

	/**
	 * Get neural in the matrix
	 * @param layer layer index
	 * @param neuralPos neural index
	 * @return neural pointer
	 * @throw invalid_argument if bounds are not correct
	 */
	std::shared_ptr<Neural>& getNeuralAt(size_t layer, size_t neuralPos);

	/**
	 * Get neural in the matrix in the last layer
	 * @param neuralPos neural index
	 * @return neural pointer
	 */
	std::shared_ptr<Neural>& getNeuralAtTheLastLayer(size_t neuralPos);

	/**
	 * Get link in the matrix
	 * @param layer layer index
	 * @param neuralPos neural index
	 * @return link pointer
	 * @throw invalid_argument if bounds are not correct
	 */
	std::shared_ptr<Link>& getLinkAt(size_t layer, size_t linkPos);

	/**
	 * Set full connection of network
	 */
	void fullConnection();

public:
	/**
	 * Constructor
	 * @param learningRate alpha value used in weight computation
	 */
	Network(double learningRate = 0.01, LossFunctionType lossFunctionType = LossFunctionType::MSE);

	/**
	 * Add input layer
	 * @param nbN number of neurals
	 * @throw logic_error if input layer is already set
	 */
	void addInputLayer(unsigned int nbN);

	/**
	 * Add hidden layer
	 * @param nbN number of neurals
	 * @param aT aggregation type
	 * @param eT excitation type
	 * @throw logic_error if input layer is not already set or output is already set
	 */
	void addLayer(unsigned int nbN, AggregationType aT, ExcitationType eT);

	/**
	 * Add last layer and set links between layers
	 * @param nbN number of neurals
	 * @param aT aggregation type
	 * @param eT excitation type
	 * @throw logic_error if input layer is not already set or output is already set
	 */
	void addOutputLayer(unsigned int nbN, AggregationType aT, ExcitationType eT);

	/**
	 * Compute learning
	 * @param inputValues input values
	 * @param expectedValues expected values
	 * @throw logic_error if input and output error are not set
	 * @throw invalid_argument if bounds are not correct
	 */
	void learn(const std::vector<double>& inputValues, const std::vector<double>& expectedValues);

	/**
	 * Compute estimation
	 * @param inputValues input values
	 * @return the max class
	 * @throw logic_error if input and output error are not set
	 * @throw invalid_argument if bounds are not correct
	 */
	int estimate(const std::vector<double>& inputValue);

	/**
	 * Get loss value
	 * @return loss value
	 */
	double getLoss();

	/**
	 * Print layers
	 * @param full if true print full matrix otherwise, just in/output
	 */
	void print(bool full = false);
};

#endif /* NETWORK_NETWORK_HPP_ */
