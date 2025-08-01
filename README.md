# Simple MLP in AssemAssemblybley

This project implements a simple feedforward Artificial Neural Network (ANN) in Assembly, capable of classifying handwritten digits from the MNIST dataset. The neural net operates on 28x28 grayscale images (flattened into 784 values) and predicts which digit (0–9) the image represents. The entire system is written in low-level Assembly, demonstrating how neural computation can be constructed from the ground up—without the abstractions of higher-level languages.

## Project Structure
The code is organized into modular components that each perform a distinct role in the neural network pipeline:

`main.s`: The entry point of the program. It initializes memory, loads the data, and controls the flow between network stages (input loading, inference, and output display).

`weights.s`: This file contains the hardcoded weight matrices for the network layers. Since dynamic training isn't the main focus in this version, weights are precomputed and stored in a linear memory layout optimized for matrix operations.

`activation.s`: Implements the activation function (e.g., sigmoid or ReLU) using Assembly routines. These functions are called during the forward pass to introduce non-linearity into the network.

`forward.s`: Contains the forward propagation logic. It performs matrix-vector multiplication between input vectors and weight matrices, applies the activation function, and computes the outputs layer-by-layer.

`input.s`: Loads and preprocesses the MNIST test data. It normalizes the image pixel values into a format usable by the network and stores them in memory buffers.

`output.s`: Takes the output layer activations and selects the neuron with the highest activation as the predicted digit. This value is then printed or stored for evaluation.

`utils.s`: Contains reusable low-level functions for tasks such as memory copying, arithmetic operations, and I/O utilities.

## How It Works
The ANN implemented here consists of an input layer (784 nodes), a single hidden layer, and an output layer (10 nodes). When a sample image is loaded, its pixel values are normalized and passed through the network. The weighted sum of inputs is computed for each layer, the activation function is applied, and the final output vector is examined to determine the predicted class.

The implementation focuses on inference—classifying already-trained data—rather than training. However, the modular architecture could be extended to support backpropagation and online weight updates in future versions.