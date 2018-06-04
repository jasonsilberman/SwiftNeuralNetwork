//
//  Layer.swift
//  SwiftNeuralNetwork
//
//  Created by Jason Silberman on 5/6/18.
//  Copyright © 2018 Jason Silberman. All rights reserved.
//

import Foundation

extension ClosedRange where Bound: FloatingPoint {
	func random() -> Bound {
		let range = self.upperBound - self.lowerBound
		let randomValue = (Bound(arc4random_uniform(UINT32_MAX)) / Bound(UINT32_MAX)) * range + self.lowerBound
		return randomValue
	}
}

func randomWeights(number: Int) -> [Double] {
	return (0..<number).map { _ in
		return (-2.0...2.0).random()
	}
}

/// A layer of the neural network.
/// Passes the inputs to each node and then collects the outputs for the next layer.
final class Layer: Codable {
	var nodes: [Node]
	var outputCache: [Double]
	
	var previousLayer: Layer?
	let numberOfNodes: Int
	
	let activationFunction: ActivationFunction
	
	init(previousLayer: Layer?, numberOfNodes: Int, activationFunction: ActivationFunction) {
		self.previousLayer = previousLayer
		self.numberOfNodes = numberOfNodes
		self.activationFunction = activationFunction
		
		self.nodes = (0..<numberOfNodes).map { _ in
			return Node(weights: randomWeights(number: previousLayer?.nodes.count ?? 0), activationFunction: activationFunction)
		}
		self.outputCache = [Double](repeating: 0, count: nodes.count)
	}
	
	func infer(inputArray: [Double]) -> [Double] {
		let inputsWithBias: [Double] = inputArray
		
		guard let _ = previousLayer else { // first layer just return inputs
			return inputArray
		}
		
		outputCache = nodes.map({ $0.infer(inputs: inputsWithBias) })
		
		return outputCache
	}
	
	func updateWeights(learningRate: Double) {
		for node in nodes {
			node.updateWeights(learningRate: learningRate)
		}
	}
	
	func calculateDeltasForOutputLayer(targetOutput: [Double]) {
		let deltas = activationFunction.computeOutputDeltas(calculatedOutputs: outputCache, targetOutputs: targetOutput)
		
		for (idx, node) in nodes.enumerated() {
			node.delta = deltas[idx]
		}
	}
	
	func calculateDeltasForHiddenLayer(nextLayer: Layer) {
		for (idx, node) in nodes.enumerated() {
			let nextWeights = nextLayer.nodes.map { $0.weights[idx] }
			let nextDeltas = nextLayer.nodes.map { $0.delta }
			let sumOfWeightsXDeltas = dotProduct(nextWeights, nextDeltas)
			node.delta = activationFunction.computeDerivative(node.cachedValue) * sumOfWeightsXDeltas
		}
	}
}
