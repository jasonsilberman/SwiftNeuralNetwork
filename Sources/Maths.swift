//
//  Maths.swift
//  SwiftNeuralNetwork
//
//  Created by Jason Silberman on 5/6/18.
//  Copyright © 2018 Jason Silberman. All rights reserved.
//

import Foundation
import Accelerate


// MARK: - SIMD Accelerated Math
// Based on example from Surge project
// https://github.com/mattt/Surge/blob/master/Sources/Surge/Arithmetic.swift

/// Find the dot product of two vectors
/// assuming that they are of the same length
/// using SIMD instructions to speed computation
func dotProduct(_ xs: [Double], _ ys: [Double]) -> Double {
	precondition(xs.count == ys.count, "Collections must have the same size")
	
	var answer: Double = 0.0
	vDSP_dotprD(xs, 1, ys, 1, &answer, vDSP_Length(xs.count))
	
	return answer
}

/// Does e^x for each value of the vector
func exp(_ xs: [Double]) -> [Double] {
	var results = [Double](repeating: 0.0, count: xs.count)
	vvexp(&results, xs, [Int32(xs.count)])
	
	return results
}

/// Subtract one vector from another
func sub(_ xs: [Double], _ ys: [Double]) -> [Double] {
	precondition(xs.count == ys.count, "Collections must have the same size")
	
	var results = [Double](ys)
	catlas_daxpby(Int32(xs.count), 1.0, xs, 1, -1, &results, 1)
	
	return results
}

/// Add one vector to another
func add(_ xs: [Double], _ ys: [Double]) -> [Double] {
	precondition(xs.count == ys.count, "Collections must have the same size")
	
	var results = [Double](ys)
	catlas_daxpby(Int32(xs.count), 1.0, xs, 1, 1, &results, 1)
	
	return results
}

/// Partial subtraction of two vectors (X - Y) leaving the rest of X
/// NOTE: Collections do not have to be the same length
func partialSub(_ xs: [Double], _ ys: [Double]) -> [Double] {
	var results = [Double](repeating: 0.0, count: xs.count)
	vDSP_vsubD(ys, 1, xs, 1, &results, 1, vDSP_Length(xs.count))
	
	return results
}

/// Multiplies two vectors
func mul(_ xs: [Double], _ ys: [Double]) -> [Double] {
	precondition(xs.count == ys.count, "Collections must have the same size")
	
	var results = [Double](repeating: 0.0, count: xs.count)
	vDSP_vmulD(xs, 1, ys, 1, &results, 1, vDSP_Length(xs.count))
	
	return results
}

/// Dividing two vectors
func div(_ xs: [Double], _ ys: [Double]) -> [Double] {
	precondition(xs.count == ys.count, "Collections must have the same size")
	
	var results = [Double](repeating: 0.0, count: xs.count)
	vDSP_vdivD(ys, 1, xs, 1, &results, 1, vDSP_Length(xs.count))
	
	return results
}

/// Sums the values of a vector
func sum(_ xs: [Double]) -> Double {
	var result: Double = 0.0
	vDSP_sveD(xs, 1, &result, vDSP_Length(xs.count))
	
	return result
}

/// Sums the squares of the values of a vector
func sumsq(_ xs: [Double]) -> Double {
	var result: Double = 0.0
	vDSP_svesqD(xs, 1, &result, vDSP_Length(xs.count))
	
	return result
}

/// Finds the average
func mean(_ xs: [Double]) -> Double {
	var result: Double = 0.0
	vDSP_meanvD(xs, 1, &result, vDSP_Length(xs.count))
	
	return result
}

/// Finds the mean square value
func measq(_ xs: [Double]) -> Double {
	var result: Double = 0.0
	vDSP_measqvD(xs, 1, &result, vDSP_Length(xs.count))
	
	return result
}

/// Finds the standard deviation
func std(_ xs: [Double]) -> Double {
	let diff = sub(xs, [Double](repeating: mean(xs), count: xs.count))
	let variance = measq(diff)
	return sqrt(variance)
}
