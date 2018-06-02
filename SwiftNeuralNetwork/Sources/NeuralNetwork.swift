//
//  NeuralNetwork.swift
//  SwiftNeuralNetwork
//
//  Created by Jason Silberman on 5/6/18.
//  Copyright © 2018 Jason Silberman. All rights reserved.
//

import Foundation

/// A feed-foward then backpropogate neural network
/// Network -> Layer -> Node
public class NeuralNetwork: Codable {
	var layers: [Layer] = []
	
	public init(layerStructure: [Int], activationFunctions: [ActivationFunction]) {
		precondition(layerStructure.count >= 3, "Layer structer must have at least three nodes (1 input, 1 hidden, 1 output)")
		precondition(layerStructure.count == activationFunctions.count, "Must supply equal number of layers and activation functions")
		
		// input layer
		layers.append(Layer(previousLayer: nil, numberOfNodes: layerStructure[0], activationFunction: activationFunctions[0]))
		
		// hidden layers
		for x in layerStructure.enumerated() where x.offset != 0 && x.offset != layerStructure.count - 1 {
			layers.append(Layer(previousLayer: layers[x.offset - 1], numberOfNodes: x.element, activationFunction: activationFunctions[x.offset]))
		}
		
		// output layer
		layers.append(Layer(previousLayer: layers[layerStructure.count - 2], numberOfNodes: layerStructure.last!, activationFunction: activationFunctions.last!))
	}
	
	/// Use neural net to infer output given some inputs
	@discardableResult
	public func infer(input: [Double]) -> [Double] {
		return layers.reduce(input) { $1.infer(inputArray: $0) }
	}
	
	/// Training procedure: infer and then backPropogate
	public func train(input: [Double], targetOutput: [Double], learningRate: Double) {
		infer(input: input)
		
		backPropogate(targetOutput: targetOutput, learningRate: learningRate)
	}
	
	/// Train with multiple sets of data
	public func train(inputs: [[Double]], targetOutputs: [[Double]], learningRate: Double) {
		let trainingData = zip(inputs, targetOutputs)
		
		for (input, targetOutput) in trainingData {
			train(input: input, targetOutput: targetOutput, learningRate: learningRate)
		}
	}
	
	/// Calculate the cost or RSS of the model
	public func rss(inputs: [[Double]], targetOutputs: [[Double]]) -> Double {
		let predictedOutputs = inputs.map({ infer(input: $0) })
		
		var errors: [Double] = []
		
		for (target, predicted) in zip(targetOutputs, predictedOutputs) {
			errors.append(contentsOf: sub(target, predicted))
		}
		
		let result = sumsq(errors)
		
		return result
	}
	
	/// After training, back propogate errors to try and fine-tune model
	private func backPropogate(targetOutput: [Double], learningRate: Double) {
		// calculate deltas for output layer
		layers.last?.calculateDeltasForOutputLayer(targetOutput: targetOutput)
		
		// calculate deltas for hidden layers
		for (idx, layer) in layers.enumerated().dropFirst().dropLast() {
			layer.calculateDeltasForHiddenLayer(nextLayer: layers[idx + 1])
		}
		
		// update weights accordingly for hidden and output layers
		for layer in layers.dropFirst() {
			layer.updateWeights(learningRate: learningRate)
		}
	}
	
}
