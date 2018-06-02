# Swift Neural Network
A neural network built in Swift.

## Usage
Below are some basic uses, you can look at [tests](https://github.com/jasonsilberman/SwiftNeuralNetwork/tree/master/Tests) to see some more advanced usages.

### Creation & Training
To get started create an instance of `NeuralNetwork` and train some data to it.

```swift
// This creates a network with three layers of the following sizes(2 nuerons, 8, 1).
// The input layer has no activation function, while the second two both use a sigmoid function.
let network = NeuralNetwork(layerStructure: [2, 8, 1], activationFunctions: [.none, .sigmoid, .sigmoid])

// We are going to train our network about the XOR operator.
let trainingData: [[Double]] = [[0, 1], [0, 0], [1, 1], [1, 0]]
let trainingResults: [[Double]] = [[1], [0], [0], [1]]

network.train(inputs: trainingData, targetOutputs: trainingResults, learningRate: 0.9)
```

### Predictions

```swift
let result = network.infer(input: [0, 1])
```

## License
This project is licensed under the [MIT License](https://github.com/jasonsilberman/SwiftNeuralNetwork/blob/master/LICENSE).

## Inspiration
All of the following projects helped contribute to my learning and creation of this project. Some of the initial code/concepts were based on them.

- https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/
- https://stackoverflow.com/a/43315365/1740273
- https://github.com/Somnibyte/MLKit
- https://github.com/davecom/SwiftSimpleNeuralNetwork
- https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
- https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
- https://github.com/jordenhill/Birdbrain
- https://github.com/mattt/Surge
- https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/
- https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions
- https://github.com/Swift-AI/NeuralNet
