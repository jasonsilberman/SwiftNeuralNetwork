//
//  DataNormalizer.swift
//  SwiftNeuralNetwork
//
//  Created by Jason Silberman on 6/3/18.
//  Copyright © 2018 Jason Silberman. All rights reserved.
//

import Foundation

/// This aligns the values of an array by its index
/// It will result in: [[a0, b0], [a1, b1]] -> [[a0, a1], [b0, b1]]
///
/// - Parameter set: The dataset to be alignd.
/// - Returns: The aligned values.
func alignByIndex<T>(set: [[T]]) -> [[T]] {
	var aligned: [[T]] = []
	for data in set {
		for (idx, value) in data.enumerated() {
			if idx >= aligned.count {
				aligned.append([])
			}
			aligned[idx].append(value)
		}
	}
	return aligned
}

/// This is a class that allows you to normalize data for use in the network.
/// It uses a z-score calculation for each data point.
public class DataNormalizer: CustomStringConvertible {
	let initialData: [[Double]]?

	let means: [Double]
	let standardDeviations: [Double]

	/// Initialize the normalizer with population means and standard deviations.
	/// The index of each should correspond to the index of the value in the array of inputs.
	///
	/// - Parameters:
	///   - means: An array of means for the data.
	///   - standardDeviations: An array of standard deviations for the data.
	public init(means: [Double], standardDeviations: [Double]) {
		self.initialData = nil
		self.means = means
		self.standardDeviations = standardDeviations
	}

	/// Initialize the normalizer with some initial data.
	/// Use this with your training data for example in order to properly calculate mean and standard deviation.
	///
	/// - Parameter initialData: The initial data that will be used to calulcate mean and standard deviation.
	public init(initialData: [[Double]]) {
		self.initialData = initialData

		let aligned = alignByIndex(set: initialData)
		self.means = aligned.map({ mean($0) })
		self.standardDeviations = aligned.map({ std($0) })
	}

	/// Normalize a set of inputs.
	///
	/// - Parameter set: An array of inputs to be normalized.
	/// - Returns: The normalized array of inputs after a z-score is calculated.
	public func normalize(dataset set: [[Double]]) -> [[Double]] {
		let normalizedSet = set.map { (data) -> [Double] in
			let normalizedData = data.enumerated().map({ (point) -> Double in
				return normalize(point: point.element, index: point.offset)
			})

			return normalizedData
		}

		return normalizedSet
	}

	/// Normalize a single input.
	///
	/// - Parameter data: The input (array of values) to be normalized.
	/// - Returns: The normalized input after a z-score is calculated.
	public func normalize(data: [Double]) -> [Double] {
		let normalizedData = data.enumerated().map({ (point) -> Double in
			return normalize(point: point.element, index: point.offset)
		})

		return normalizedData
	}

	/// Normalizes a single value.
	///
	/// - Parameters:
	///   - point: The value to be normalized.
	///   - index: The index of the value in the input array.
	/// - Returns: The noramlized value after a z-score is calculated.
	public func normalize(point: Double, index: Int) -> Double {
		let normalizedPoint = (point - means[index]) / standardDeviations[index]

		return normalizedPoint
	}

	public var description: String {
		return "DataNormalizer(means: \(means), standardDeviations: \(standardDeviations))"
	}
}
