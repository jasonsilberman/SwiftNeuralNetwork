//
//  XORTests.swift
//  SwiftNeuralNetworkTests
//
//  Created by Jason Silberman on 6/1/18.
//  Copyright © 2018 Jason Silberman. All rights reserved.
//

import XCTest
@testable import SwiftNeuralNetwork

/// This is a test of XOR (exclusive or). We have a neural network that will take 2 inputs and produce 1 output.
/// There is also a hidden layer with 20 neurons that is used to help predict the output. The 20 is chosen pretty randomly. I just played with it until it felt like it was consistanyl taking less iterations to train.
/// In `setup()` the network is trained to a RSS of 0.02 (this is just training the network until it is very accurate). We give the network 2000 tries to achieve this. Usually it happens in ~500 iterations.
class XORTests: XCTestCase {
	
	struct Constants {
		static let learningRate: Double = 0.9
	}
	
	var network: NeuralNetwork!
    
    override func setUp() {
        super.setUp()
		network = NeuralNetwork(layerStructure: [2, 20, 1], activationFunctions: [.none, .sigmoid, .sigmoid])
		
		let trainingData: [[Double]] = [[0, 1], [0, 0], [1, 1], [1, 0]]
		let trainingResults: [[Double]] = [[1], [0], [0], [1]]
		
		var lastRSS: Double = 1
		let threshold: Double = 0.02
		var count = 0
		let maxIterations = 2000
		
		while lastRSS > threshold && count < maxIterations {
			count += 1
			
			network.train(inputs: trainingData, targetOutputs: trainingResults, learningRate: Constants.learningRate)
			
			lastRSS = network.rss(inputs: trainingData, targetOutputs: trainingResults)
		}
		
		if lastRSS > threshold {
			XCTFail("FAILED! Could not achieve an RSS of: \(threshold). Final RSS was: \(lastRSS)")
		} else {
			print("SUCCESSFULLY reached RSS of \(lastRSS) after \(count) tries.")
		}
    }
    
    func testXOR() {
        let validationData: [[Double]] = [[0, 1], [0, 0], [1, 1], [1, 0]]
		let validationResults: [[Double]] = [[1], [0], [0], [1]]
		
		for (input, expectedResult) in zip(validationData, validationResults) {
			let actualResult = network.infer(input: input)
			// if difference is within .5 then the correct answer was given
			let difference = abs(actualResult[0] - expectedResult[0])
			XCTAssertTrue(difference < 0.5, "Network validation failed. Input: \(input). Expected: \(expectedResult). Actual: \(actualResult)")
		}
    }
    
}
