//
//  SineTests.swift
//  SwiftNeuralNetworkTests
//
//  Created by Jason Silberman on 6/3/18.
//  Copyright © 2018 Jason Silberman. All rights reserved.
//

import XCTest
@testable import SwiftNeuralNetwork

func generateData(count: Int) -> (xs: [[Double]], ys: [[Double]]) {
	let randoms = (0..<count).map { _ in
		return (-4...4).random()
	}
	let xs = randoms.map { [$0] }
	let ys = randoms.map({ [(0.5 * sin($0)) + 0.5] })
	
	return (xs, ys)
}

/// This gives an approximation of the y = 0.5sin(x) + 0.5 function.
class SineTests: XCTestCase {
	struct Constants {
		static let learningRate: Double = 0.9
	}
	
	var network: NeuralNetwork!
	
	var normalizer: DataNormalizer!
	
	override func setUp() {
		super.setUp()
		network = NeuralNetwork(layerStructure: [1, 12, 1], activationFunctions: [.none, .sigmoid, .sigmoid])
		
		let (unnormalizedData, trainingResults) = generateData(count: 10)
		
		normalizer = DataNormalizer(initialData: unnormalizedData)
		let trainingData = normalizer.normalize(dataset: unnormalizedData)
		
		print(trainingData)
		print(trainingResults)
		
		var lastRSS: Double = 1
		let threshold: Double = 0.02
		var count = 0
		let maxIterations = 10000
		
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
	
	func testSine() {
		let validationData: [[Double]] = [[.pi/2]]
		let validationResults: [[Double]] = [[1]]
		
		for (input, expectedResult) in zip(validationData, validationResults) {
			let actualResult = network.infer(input: input)
			// if difference is within 0.5 then the correct answer was given
			print(input, expectedResult, actualResult)
//			let difference = abs(actualResult[0] - expectedResult[0])
//			XCTAssertTrue(difference < 0.5, "Network validation failed. Input: \(input). Expected: \(expectedResult). Actual: \(actualResult)")
		}
	}
}
