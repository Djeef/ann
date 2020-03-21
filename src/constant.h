/*
 * constant.h
 *
 *  Created on: Mar 8, 2018
 *      Author: Jean-François Erdelyi
 */

#pragma once

// Aggragation methods
enum class AggregationType {
	NO_AGG, SUM
};

// Neural excitation methods
enum class ExcitationType {
	NO_EXC, SIGMOID, STEP, TAN_HYP
};

// Loss function
enum class LossFunctionType {
	MSE
};

// Neural types
enum class NeuralType {
	INPUT_NEURAL, OUTPUT_NEURAL, HIDDEN_NEURAL, BIAIS_NEURAL
};

// Connect direction
enum class ConnectDirection {
	FORWARD, BACKWARD, BOTH
};
