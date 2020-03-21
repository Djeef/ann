/*
 * main.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Jean-François Erdelyi
 */

#include <iostream>
#include "constant.h"
#include "Network/Network.hpp"

using namespace std;

int main() {
	// Build ANN
	Network n = Network(0.1);	
	n.addInputLayer(3);
	n.addLayer(3, AggregationType::SUM, ExcitationType::SIGMOID);
	
	n.addLayer(4, AggregationType::SUM, ExcitationType::SIGMOID);
	n.addLayer(4, AggregationType::SUM, ExcitationType::SIGMOID);
	
	n.addLayer(2, AggregationType::SUM, ExcitationType::SIGMOID);
	n.addOutputLayer(2, AggregationType::SUM, ExcitationType::SIGMOID);

	// Learn
	int j = 0;
	cout << "LEARN" << endl;
	
	vector<double> vectorsI(3);
	vector<double> vectorsE(2);

	double input1 = 0;
	double input2 = 0;
	double input3 = 0;

	double expValue;
	do {
		input1 = (rand() % 100) / 100.0;     
		input2 = (rand() % 100) / 100.0;
		input3 = (rand() % 100) / 100.0;

		if (input1 < 0.5) {
			expValue = 0;
		} else {
			expValue = 1;
		}

		vectorsI.at(0) = input1;
		vectorsI.at(1) = input2;
		vectorsI.at(2) = input3;

		vectorsE.at(0) = expValue;
		vectorsE.at(1) = 1 - expValue;
		n.learn(vectorsI, vectorsE);
		j++;
	} while (n.getLoss() > 0.01);
	cout << j << endl;

	vector<double> testValue(3);
	int ok = 0;
	int nbTry = 1000;
	for (int i = 0; i < nbTry; ++i) {
		input1 = (rand() % 100) / 100.0;
		input2 = (rand() % 100) / 100.0;
		input3 = (rand() % 100) / 100.0;		
		
		testValue.at(0) = input1;
		testValue.at(1) = input2;
		testValue.at(2) = input3;

		if (input1 < 0.5 && n.estimate(testValue) == 1) {
			ok++;
		} else if (input1 >= 0.5 && n.estimate(testValue) == 0) {
			ok++;
		}
	}

	n.print();
	cout << "Sucess rate: "  << (ok / (double)nbTry) * 100 << "%" << endl;
	return 0;
}
