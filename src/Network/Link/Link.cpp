/**
 * Link.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Jean-François Erdelyi
 */

#include "Link.hpp"

// Learning data
double Link::learningRate = 0.0;

Link::Link(std::shared_ptr<Neural> previousNeural, std::shared_ptr<Neural> nextNeural) {
	this->previousNeural = previousNeural;
	this->nextNeural = nextNeural;

	// Random weight
	weight = rand() / RAND_MAX;
}

void Link::connect(ConnectDirection direction) {
	// Connect link depending of the direction
	switch (direction) {
		case ConnectDirection::FORWARD:
			nextNeural->addInputLink(shared_from_this());
			break;
		case ConnectDirection::BACKWARD:
			previousNeural->addOutputLink(shared_from_this());
			break;
		case ConnectDirection::BOTH:
			nextNeural->addInputLink(shared_from_this());
			previousNeural->addOutputLink(shared_from_this());
			break;
	}
}

double Link::getComputedValue() {
	return previousNeural->getOutputValue() * weight;
}

double Link::getComputedDelta() {
	return nextNeural->getDelta() * weight;
}

double Link::getPreviousValue() {
	return previousNeural->getOutputValue();
}

double Link::getNextDelta() {
	return nextNeural->getDelta();
}

void Link::computeWeight() {
	// w = w - alpha * J'(w) 
	// J'(w) = z * delta
	weight = weight - learningRate * getPreviousValue() * getNextDelta();
}

void Link::setLearningRate(double learningRate) {
	Link::learningRate = learningRate;
}

std::string Link::toString() {
	return std::to_string(weight);
}
