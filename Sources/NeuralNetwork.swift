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
public final class NeuralNetwork: Codable {
	var layers: [Layer] = []
	
	/// Initialize a neural network.
	///
	/// - Parameters:
	///   - layerStructure: Each number in the layoutStructure represent a layer and the value is the number of nodes in said layer.
	///   - activationFunctions: A list of activation functions for each layer. There must be an activation function for each layer.
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
	
	/// Use the neural network to infer output given some inputs.
	///
	/// - Parameter input: The data that should be run through the neural network.
	/// - Returns: The output returned after feeding the inputs into the network.
	@discardableResult
	public func infer(input: [Double]) -> [Double] {
		return layers.reduce(input) { $1.infer(inputArray: $0) }
	}
	
	/// Train a single input and output.
	/// This will first infer the output after feeding the input through the network,
	/// and then it will update the internal weights based on the error between the actual and the target results.
	///
	/// - Parameters:
	///   - input: The data that should be run through the neural network.
	///   - targetOutput: The output that the neural network this produce.
	///   - learningRate: The rate at which the network learns. Typically between 0.1 to 0.9.
	public func train(input: [Double], targetOutput: [Double], learningRate: Double) {
		infer(input: input)
		
		backPropogate(targetOutput: targetOutput, learningRate: learningRate)
	}
	
	/// Train a a set of inputs and outputs.
	/// This will first infer the output after feeding the input through the network,
	/// and then it will update the internal weights based on the error between the actual and the target results.
	///
	/// - Parameters:
	///   - inputs: The data set that should be run through the neural network.
	///   - targetOutputs: The output set that the neural network this produce.
	///   - learningRate: The rate at which the network learns. Typically between 0.1 to 0.9.
	public func train(inputs: [[Double]], targetOutputs: [[Double]], learningRate: Double) {
		let trainingData = zip(inputs, targetOutputs)
		
		for (input, targetOutput) in trainingData {
			train(input: input, targetOutput: targetOutput, learningRate: learningRate)
		}
	}
	
	/// This will calculate the cost or residual sum of squares (RSS) of the model.
	/// This function will compare the actual and target results and calculate the error between the two.
	///
	/// - Parameters:
	///   - inputs: The data set that should be run through the neural network.
	///   - targetOutputs: The output set that the neural network this produce.
	/// - Returns: The calculated error of the model.
	public func rss(inputs: [[Double]], targetOutputs: [[Double]]) -> Double {
		let predictedOutputs = inputs.map({ infer(input: $0) })
		
		var errors: [Double] = []
		
		for (target, predicted) in zip(targetOutputs, predictedOutputs) {
			errors.append(contentsOf: sub(target, predicted))
		}
		
		let result = sumsq(errors)
		
		return result
	}
	
	/// After training, back propogate errors to update the weights in order to fine-tune model.
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
