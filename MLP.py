# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the Iris dataset from scikit-learn
iris = load_iris()

# Select the first 100 samples and only the first 2 features (sepal length and sepal width)
# This subset corresponds to a binary classification problem (two classes)
X = iris.data[:100, :2]  # Features: first two columns
y = iris.target[:100]    # Labels: target classes for first 100 samples

# Split the data into training and testing sets
# 80% for training and 20% for testing to evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def build_model(act, opt):
    """
    Builds and compiles a simple neural network model.
    
    Parameters:
    act (str): The activation function to be used in the hidden layer.
    opt (tf.keras.optimizers.Optimizer): The optimizer for compiling the model.
    
    Returns:
    model (tf.keras.models.Sequential): The compiled Keras model ready for training.
    """
    # Create a Sequential model instance
    model = Sequential([
        # Add a hidden layer with 10 neurons, specified activation function,
        # and input shape matching the number of features in the training data.
        Dense(10, activation=act, input_shape=(X_train.shape[1],)),
        
        # Add an output layer with 1 neuron and sigmoid activation function
        # which is appropriate for binary classification.
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with:
    # - The specified optimizer (opt)
    # - Binary crossentropy loss (since it's a binary classification problem)
    # - Accuracy as a metric to evaluate the performance during training
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Return the compiled model
    return model

# Iterate over different activation functions and optimizers
# Test combinations of activation functions: 'sigmoid', 'tanh', 'relu'
# And optimizers: Adam and SGD
for act in ['sigmoid', 'tanh', 'relu']:
    for opt_name, opt in [('Adam', Adam()), ('SGD', SGD())]:
        # Build the model with the current activation function and optimizer
        model = build_model(act, opt)
        
        # Train the model on the training data
        # Using 50 epochs and a batch size of 16, with verbose set to 0 to hide training logs
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        
        # Evaluate the trained model on the test data
        # This returns the loss and accuracy of the model
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Print the results with a clear description of the configuration and achieved accuracy
        print(f"The model with activation function '{act}' and optimizer '{opt_name}' achieved an accuracy of {acc * 100:.2f}%")
