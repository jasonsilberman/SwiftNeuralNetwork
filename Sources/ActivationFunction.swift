//
//  ActivationFunction.swift
//  SwiftNeuralNetwork
//
//  Created by Jason Silberman on 5/6/18.
//  Copyright © 2018 Jason Silberman. All rights reserved.
//

import Foundation

/// The activation function to be used in a node.
/// NOTE: Do not use softmax yet as it is not implemented.
public enum ActivationFunction: String, Equatable, Codable {
	case none, sigmoid, leakyReLU, softmax
	
	func computeActivation(_ x: Double) -> Double {
		switch self {
		case .none:
			return x
		case .sigmoid:
			return sigmoidActivation(x)
		case .leakyReLU:
			return leakyReLUActivation(x)
		default:
			return 0.0
		}
	}
	
	func computeDerivative(_ x: Double) -> Double {
		switch self {
		case .none:
			return 0.0
		case .sigmoid:
			return sigmoidDerivative(x)
		case .leakyReLU:
			return leakyReLUDerivative(x)
		default:
			return 0.0
		}
	}
	
	func computeOutputs(_ xs: [Double]) -> [Double] {
		switch self {
		case .softmax:
			return softmaxActivations(xs)
		default:
			return xs.map({ computeActivation($0) })
		}
	}
	
	func computeOutputDeltas(calculatedOutputs: [Double], targetOutputs: [Double]) -> [Double] {
		switch self {
		case .sigmoid:
			return zip(calculatedOutputs, targetOutputs).map({ sigmoidDerivative($0) * ($1 - $0) })
		case .softmax:
			return partialSub(targetOutputs, calculatedOutputs)
		default:
			return []
		}
	}
	
	// MARK: - Functions
	
	private func sigmoidActivation(_ x: Double) -> Double {
		return 1 / (1 + exp(-x))
	}
	
	private func sigmoidDerivative(_ x: Double) -> Double {
		return x * (1 - x)
	}
	
	private func leakyReLUActivation(_ x: Double) -> Double {
		return max(0.01*x, x)
	}
	
	private func leakyReLUDerivative(_ x: Double) -> Double {
		return (x < 0.0) ? 0.01 : 1.0
	}
	
	private func softmaxActivations(_ xs: [Double]) -> [Double] {
		let exponentials = exp(xs)
		let total = sum(exponentials)
		
		return div(exponentials, [Double](repeating: total, count: exponentials.count))
	}
}
