//
//  DataNormalizerTests.swift
//  SwiftNeuralNetworkTests
//
//  Created by Jason Silberman on 6/3/18.
//  Copyright © 2018 Jason Silberman. All rights reserved.
//

import XCTest
@testable import SwiftNeuralNetwork

class DataNormalizerTestsTests: XCTestCase {
	func testAlignByIndex() {
		let first = [0, 1, 2, 3]
		let second = [10, 11, 12, 13]
		let merged = zip(first, second).map { [$0.0, $0.1] }
		let aligned = alignByIndex(set: merged)
		
		XCTAssertEqual(aligned[0], first, "Did not correctly align first array")
		XCTAssertEqual(aligned[1], second, "Did not correctly align second array")
	}
	
	func testBasicNormalization() {
		let xs: [[Double]] = [[0, 0], [10, 1], [20, 2], [30, 3], [40, 4]]
		let normalizer = DataNormalizer(initialData: xs)
		
		let testData: [Double] = [20, 2] // this is the mean point so z-scores should both be zero
		XCTAssertEqual(normalizer.normalize(data: testData), [0, 0], "Did not correctly normalize dataset")
	}
}
