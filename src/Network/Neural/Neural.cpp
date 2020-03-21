/*
 * Neural.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Jean-François Erdelyi
 */
#include <iostream>

#include "Neural.hpp"

#include "../Functions/Aggregation/Sum/Sum.hpp"
#include "../Functions/Excitation/Sigmoid/Sigmoid.hpp"
#include "../Functions/Excitation/Step/Step.hpp"
#include "../Functions/Excitation/TanHyp/TanHyp.hpp"

Neural::Neural(NeuralType t, AggregationType aT, ExcitationType eT, double v) {
	type = t;
	inputValue = 0.0;
	if (type == NeuralType::BIAIS_NEURAL) {
		outputValue = v;
	} else {
		outputValue = 0.0;
	}
	delta = 0.0;
	nextLayerComputedDelta = 0.0;
	
	// Create aggregation
	switch (aT) {
		case AggregationType::NO_AGG:
			aggregationFunction = nullptr;
			break;
		case AggregationType::SUM:
			aggregationFunction = std::make_unique<Sum>();
			break;
	}

	// Create excitation
	switch (eT) {
		case ExcitationType::NO_EXC:
			excitationFunction = nullptr;
			break;
		case ExcitationType::SIGMOID:
			excitationFunction = std::make_unique<Sigmoid>();
			break;
		case ExcitationType::STEP:
			excitationFunction = std::make_unique<Step>();
			break;
		case ExcitationType::TAN_HYP:
			excitationFunction = std::make_unique<TanHyp>();
			break;
	}
}

void Neural::addBias(double value) {
	// Build the bias neural and link it to the current neural (if is not input neural)
	if (type == NeuralType::HIDDEN_NEURAL || type == NeuralType::OUTPUT_NEURAL) {
		biasNeural = std::make_shared<Neural>(NeuralType::BIAIS_NEURAL, AggregationType::NO_AGG, ExcitationType::NO_EXC, value);
		biasLink = connectFrom(biasNeural);
	} else {
		biasNeural = nullptr;
		biasLink = nullptr;
	}
}

void Neural::addInputLink(std::shared_ptr<Link> link) {
	if (type == NeuralType::BIAIS_NEURAL) {
		throw std::logic_error("Bias neural have no input links");
	}
	if (type == NeuralType::INPUT_NEURAL) {
		throw std::logic_error("Input neural have no input links");
	}

	inputLinks.push_back(link);
}

void Neural::addOutputLink(std::shared_ptr<Link> link) {
	if (type == NeuralType::OUTPUT_NEURAL) {
		throw std::logic_error("Output neural have no output links");
	}

	outputLinks.push_back(link);
}

void Neural::forwardPropagation() {
	if (type == NeuralType::BIAIS_NEURAL) {
		throw std::logic_error("Bias neural can not be computed");
	}

	// Compute aggregation
	if (aggregationFunction && inputLinks.size()) {
		inputValue = aggregationFunction->compute(inputLinks);
	}

	// Compute excitation
	if (excitationFunction) {
		outputValue = excitationFunction->compute(inputValue);
	}
}

void Neural::backwardPropagation() {
	if (type == NeuralType::BIAIS_NEURAL) {
		throw std::logic_error("Bias neural can not be computed");
	}
	if (type == NeuralType::INPUT_NEURAL) {
		throw std::logic_error("Input neural can not be computed");
	}

	// Compute delta
	setDelta();

	// Propagate to input links
	computeWeight();
}

void Neural::setDelta()  {
	// Sum of links errors
	if (type == NeuralType::HIDDEN_NEURAL) {
		nextLayerComputedDelta = 0.0;
		for (size_t i = 0; i < outputLinks.size(); ++i) {
			nextLayerComputedDelta += outputLinks.at(i)->getComputedDelta();
		}
	} 

	// Compute delta f'(value) * sum(delta * w)[n+1]
	delta = excitationFunction->derivative(inputValue) * nextLayerComputedDelta;
}

double Neural::computeDeltaOutput(double expectedValue) {
	if (type == NeuralType::OUTPUT_NEURAL) {
		nextLayerComputedDelta = outputValue - expectedValue;
	} else {
		throw std::logic_error("Only output layer can compute delta output");
	}
	return nextLayerComputedDelta;
}

void Neural::computeWeight() {
	// Compute new weight of all input links
	for (auto link : inputLinks) {
		link->computeWeight();
	}
}

std::shared_ptr<Link>& Neural::connectTo(std::shared_ptr<Neural>& next) {
	auto link = std::make_shared<Link>(shared_from_this(), next);
	link->connect(ConnectDirection::FORWARD);
	outputLinks.push_back(link);
	return outputLinks.back();
}

std::shared_ptr<Link>& Neural::connectFrom(std::shared_ptr<Neural>& previous) {
	auto link = std::make_shared<Link>(previous, shared_from_this());
	link->connect(ConnectDirection::BACKWARD);
	inputLinks.push_back(link);
	return inputLinks.back();
}

double Neural::getDelta() {
	return this->delta;
}

double Neural::getOutputValue() {
	return this->outputValue;
}

std::shared_ptr<Neural>& Neural::getBiasNeural() {
	return biasNeural;
}

std::shared_ptr<Link>& Neural::getBiasLink() {
	return biasLink;
}

void Neural::setValue(double value) {
	if (type == NeuralType::INPUT_NEURAL) {
		this->outputValue = value;
	} else {
		throw std::logic_error("Only input layer can be set");
	}
}

std::string Neural::toString() {
	// Return bias value if is present
	if (!biasLink) {
		return std::to_string(outputValue);
	}
	return std::to_string(outputValue) + "(" + std::to_string(biasLink->getComputedValue()) + ")";
}
