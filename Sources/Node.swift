//
//  Node.swift
//  SwiftNeuralNetwork
//
//  Created by Jason Silberman on 5/6/18.
//  Copyright © 2018 Jason Silberman. All rights reserved.
//

import Foundation

/// A single node in a layer
final class Node: Codable {
	var weights: [Double]
	var cachedValue: Double = 0.0
	var cachedInput: [Double]
	var delta: Double = 0.0
	
	let activationFunction: ActivationFunction
	
	init(weights: [Double], activationFunction: ActivationFunction) {
		self.weights = weights
		self.activationFunction = activationFunction
		self.cachedInput = [Double](repeating: 0, count: weights.count)
	}
	
	func infer(inputs: [Double]) -> Double {
		cachedInput = inputs
		let sum = dotProduct(inputs, weights)
		cachedValue = activationFunction.computeActivation(sum)
		return cachedValue
	}
	
	func updateWeights(learningRate: Double) {
		for (idx, weight) in weights.enumerated() {
			weights[idx] = weight + (learningRate * cachedInput[idx] * delta)
		}
	}
}
