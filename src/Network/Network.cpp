/*
 * Network.cpp
 *
 *  Created on: Mar 6, 2018
 *      Author: Jean-François Erdelyi
 */

#include <iostream>
#include <ctime>

#include "Network.hpp"
#include "Functions/Loss/Mse/Mse.hpp"

Network::Network(double learningRate, LossFunctionType lossFunctionType) {
	Link::setLearningRate(learningRate);

	// Input layer is not set
	isInputSet = false;

	// Output layer is not set
	isOutputSet = false;

	// The number of neurals
	neuralsCount = 0;

	// The number of links
	linksCount = 0;

	// The number of output neurals
	outputsCount = 0;

	// The number of layers
	lastNeuralLayerIndex = 0;
	
	// Loss error used for back propagation
	loss = 1.0;

	// Instanciation of the loss function
	if (lossFunctionType == LossFunctionType::MSE) {
		this->lossFunction = std::make_unique<Mse>();
	}

	// Init random
	srand((int) time(NULL));
}

void Network::addLayer(NeuralType t, unsigned int nbN, AggregationType aT, ExcitationType eT) {
	// Add a layer
	std::vector<std::shared_ptr<Neural>> layer = std::vector<std::shared_ptr<Neural>>();
	for (unsigned int i = 0; i < nbN; ++i) {
		// New neural define by:
		// Type, aggregation and excitation
		auto neural = std::make_shared<Neural>(t, aT, eT);
		layer.push_back(neural);
		neural->addBias(-1.0);
	}
	neurals.push_back(layer);
	
	// Update the number of neurals
	neuralsCount += nbN;
}

void Network::addInputLayer(unsigned int nbN) {
	// Add input layer (no aggregation and no excitation)
	if (!isInputSet) {
		addLayer(NeuralType::INPUT_NEURAL, nbN, AggregationType::NO_AGG, ExcitationType::NO_EXC);
		isInputSet = true;
	} else {
		throw std::logic_error("Input layer is already set");
	}
}

void Network::addLayer(unsigned int nbN, AggregationType aT, ExcitationType eT) {
	// Add hidden layer
	if (isInputSet && !isOutputSet) {
		addLayer(NeuralType::HIDDEN_NEURAL, nbN, aT, eT);
	} else {
		if (!isInputSet) {
			throw std::logic_error("Input layer is not set yet");
		} else if (isOutputSet) {
			throw std::logic_error("Output layer is already set");
		}
	}
}

void Network::addOutputLayer(unsigned int nbN, AggregationType aT, ExcitationType eT) {
	// Add output layer
	if (isInputSet && !isOutputSet) {
		addLayer(NeuralType::OUTPUT_NEURAL, nbN, aT, eT);
		isOutputSet = true;
	}
	else {
		if (!isInputSet) {
			throw std::logic_error("Input layer is not set yet");
		}
		else if (isOutputSet) {
			throw std::logic_error("Output layer is already set");
		}
	}

	// TODO: implements other kind of interconnection
	fullConnection();
}

void Network::fullConnection() {
	// Add counts output and layers
	outputsCount = (unsigned int)neurals.back().size();
	lastNeuralLayerIndex = (unsigned int)(neurals.size() - 1);

	// Connect all neurals
	for (size_t i = 0; i < lastNeuralLayerIndex; ++i) {
		size_t currentLayerSize = neurals.at(i).size();
		size_t nextLayerSize = neurals.at(i + 1).size();

		// For each layer (l) to the next one (l + 1)
		std::vector<std::shared_ptr<Link>> layer = std::vector<std::shared_ptr<Link>>();
		for (size_t j = 0; j < currentLayerSize; ++j) {
			for (size_t k = 0; k < nextLayerSize; ++k) {
				// For each cell in the current layer
				// Connect output link to the input link of each next neural's layer
				std::shared_ptr<Neural> previous = getNeuralAt(i, j);
				std::shared_ptr<Neural> next = getNeuralAt(i + 1, k);

				// Create new link
				auto link = std::make_shared<Link>(previous, next);
				link->connect(ConnectDirection::BOTH);
				layer.push_back(link);
			}
		}

		// Add bias links (hidden or output neurals)
		links.push_back(layer);

		// Update the number of links
		linksCount += (unsigned int)(currentLayerSize * nextLayerSize);
	}
}

std::shared_ptr<Neural>& Network::getNeuralAt(size_t layer, size_t neuralPos) {
	// Check the validity of parameters
	if (neurals.size() < layer) {
		throw std::invalid_argument("Layer is out of bound");
	}
	if (neurals.at(layer).size() < neuralPos) {
		throw std::invalid_argument("Neural position is out of bound");
	}

	// Return the neural
	return neurals.at(layer).at(neuralPos);
}

std::shared_ptr<Neural>& Network::getNeuralAtTheLastLayer(size_t neuralPos) {
	// Return the neural
	return getNeuralAt(lastNeuralLayerIndex, neuralPos);
}

std::shared_ptr<Link>& Network::getLinkAt(size_t layer, size_t linkPos) {
	// Check the validity of parameters
	if (links.size() < layer) {
		throw std::invalid_argument("Layer is out of bound");
	}
	if (links.at(layer).size() < linkPos) {
		throw std::invalid_argument("Link position is out of bound");
	}

	// Return the link
	return links.at(layer).at(linkPos);
}

int Network::getClassMax() {
	int maxClass = 0;
	double maxClassValue = getNeuralAtTheLastLayer(0)->getOutputValue();

	// For each output neurals, get the highest neural
	for (unsigned int i = 1; i < outputsCount; ++i) {
		std::shared_ptr<Neural> n = getNeuralAtTheLastLayer(i);
		if (n->getOutputValue() > maxClassValue) {
			maxClass = i;
		}
	}
	return maxClass;
}

void Network::learn(const std::vector<double>& inputValues, const std::vector<double>& expectedValues) {
	// Check if the network is ready
	if (!isInputSet) {
		throw std::logic_error("Input layer is not set yet");
	}
	if (!isOutputSet) {
		throw std::logic_error("Output layer is not set yet");
	}

	// Control input parameter: input values
	if (inputValues.size() < neurals.at(0).size()) {
		throw std::invalid_argument("The input values vector should be, at least, equals to input layer");
	}
	// Control input parameter: expected values
	if (expectedValues.size() < neurals.at(lastNeuralLayerIndex).size()) {
		throw std::invalid_argument("The expected values vector should be, at least, equals to output layer");
	}

	// Set inputs
	for (size_t i = 0; i < neurals.at(0).size(); ++i) {
		getNeuralAt(0, i)->setValue(inputValues.at(i));
	}

	// Propagate
	for (size_t i = 0; i < neurals.size(); ++i) {
		for (size_t j = 0; j < neurals.at(i).size(); ++j) {
			getNeuralAt(i, j)->forwardPropagation();
		}
	}

	// Get output values
	std::vector<double> outputErrors;
	for (size_t j = 0; j < neurals.at(lastNeuralLayerIndex).size(); ++j) {
		outputErrors.push_back(getNeuralAt(lastNeuralLayerIndex, j)->computeDeltaOutput(expectedValues.at(j)));
	}

	// Compare with expected values (initialisation of the back propagation)
	this->loss = this->lossFunction->compute(outputErrors);

	// Back propagation
	backPropagation();
}

int Network::estimate(const std::vector<double>& inputValue) {
	// Check if the network is ready
	if (!isInputSet) {
		throw std::logic_error("Input layer is not set yet");
	}
	if (!isOutputSet) {
		throw std::logic_error("Output layer is not set yet");
	}

	// Control input parameter: input values
	if (inputValue.size() < neurals.at(0).size()) {
		throw std::invalid_argument("The input values vector should be, at least, equals to input layer");
	}

	// Set inputs
	for (size_t i = 0; i < neurals.at(0).size(); ++i) {
		getNeuralAt(0, i)->setValue(inputValue.at(i));
	}

	// Propagate
	for (size_t i = 0; i < neurals.size(); ++i) {
		for (size_t j = 0; j < neurals.at(i).size(); ++j) {
			getNeuralAt(i, j)->forwardPropagation();
		}
	}

	// Return max class
	return getClassMax();
}

double Network::getLoss() {
	return loss;
}

void Network::backPropagation() {
	// Back propagation (hidden/output layers only: not input layer)
	for (size_t i = lastNeuralLayerIndex; i >= 1; --i) {
		// Every neurals of the layer
		for (size_t j = 0; j < neurals.at(i).size(); ++j) {
			getNeuralAt(i, j)->backwardPropagation();
		}
	}
}

void Network::print(bool full) {
	if (full) {
		// Full network

		for (size_t i = 0; i < links.size(); ++i) {
			if (isInputSet && i == 0) {
				std::cout << "INPUT LAYER:\t";
				for (size_t j = 0; j < neurals.at(0).size(); ++j) {
					std::cout << getNeuralAt(0, j)->toString() << " ";
				}
				std::cout << std::endl;
			}
			else {
				std::cout << "NEURALS LAYER:\t";
				for (size_t j = 0; j < neurals.at(i).size(); ++j) {
					std::cout << getNeuralAt(i, j)->toString() << " ";
				}
				std::cout << std::endl;
			}

			std::cout << std::endl << "LINKS LAYER:\t";
			for (size_t j = 0; j < links.at(i).size(); ++j) {
				std::cout << getLinkAt(i, j)->toString() << " ";
			}
			std::cout << std::endl << std::endl;
		}

		if (isOutputSet) {
			std::cout << "OUTPUT LAYER:\t";
			for (size_t j = 0; j < neurals.at(lastNeuralLayerIndex).size(); ++j) {
				std::cout << getNeuralAt(lastNeuralLayerIndex, j)->toString() << " ";
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
		std::cout << neuralsCount << " neurals and " << linksCount << " links (" << loss << ")" << std::endl;
		std::cout << std::endl;
	}
	else {
		// Input and output only

		if (isInputSet) {
			std::cout << "INPUT LAYER:\t";
			for (size_t j = 0; j < neurals.at(0).size(); ++j) {
				std::cout << getNeuralAt(0, j)->toString() << " ";
			}
			std::cout << std::endl;
		}
		if (isOutputSet) {
			std::cout << "OUTPUT LAYER:\t";
			for (size_t j = 0; j < neurals.at(lastNeuralLayerIndex).size(); ++j) {
				std::cout << getNeuralAt(lastNeuralLayerIndex, j)->toString() << " ";
			}
			std::cout << std::endl;
		}
	}
}
