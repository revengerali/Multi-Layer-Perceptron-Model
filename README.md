# Multi-Layer-Perceptron-Model


Neural Network Experimentation with the Iris Dataset
This repository contains a Python script that demonstrates how to build, train, and evaluate a simple neural network for binary classification using a subset of the Iris dataset.

Overview
Dataset Preparation:
The script loads the Iris dataset from scikit-learn, selects the first 100 samples (binary classification), and uses only the first two features. The data is then split into training and testing sets.

Model Building:
A function (build_model) creates a Sequential neural network using TensorFlow Keras. The network comprises:

A hidden layer with 10 neurons that uses a specified activation function (sigmoid, tanh, or relu).
An output layer with 1 neuron and a sigmoid activation function, suitable for binary classification.
Experimentation:
The script iterates through different combinations of activation functions and optimizers (Adam and SGD). For each configuration, the model is trained for 50 epochs and evaluated on the test data, with the accuracy results printed out.

Learning Resource:
Detailed comments are included throughout the script to explain each step, making this repository an excellent learning tool for those interested in neural network configuration and experimentation.
