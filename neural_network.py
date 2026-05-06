"""
Neural Network Module for TFML Assignment 1
============================================
Implements a fully-connected feedforward neural network from scratch using NumPy.
Architecture: 64-X-3 (input-hidden-output) with sigmoid activation and bias.

The network is trained using backpropagation with gradient descent.
This from-scratch implementation is appropriate for a Theoretical Foundations of ML course,
demonstrating understanding of the underlying mathematics.
"""

import numpy as np
import os


class NeuralNetwork:
    """
    A single hidden-layer feedforward neural network.
    
    Architecture: input_size -> hidden_size -> output_size
    Activation: Sigmoid for both hidden and output layers
    Loss: Mean Squared Error (MSE)
    Optimizer: Gradient Descent with configurable learning rate
    
    Attributes:
        input_size (int): Number of input features (64 for 8x8 images).
        hidden_size (int): Number of hidden layer neurons.
        output_size (int): Number of output neurons (3 for B, 0, E).
        W1 (np.ndarray): Input-to-hidden weight matrix (input_size x hidden_size).
        b1 (np.ndarray): Hidden layer bias vector (1 x hidden_size).
        W2 (np.ndarray): Hidden-to-output weight matrix (hidden_size x output_size).
        b2 (np.ndarray): Output layer bias vector (1 x output_size).
    """
    
    def __init__(self, input_size=64, hidden_size=3, output_size=3, seed=42):
        """
        Initialize the neural network with random weights.
        
        Uses Xavier/Glorot initialization for better convergence:
        weights are drawn from N(0, sqrt(2 / (fan_in + fan_out)))
        
        Args:
            input_size (int): Dimension of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons.
            seed (int): Random seed for reproducibility.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        np.random.seed(seed)
        
        # Xavier initialization for input-to-hidden weights
        limit1 = np.sqrt(2.0 / (input_size + hidden_size))
        self.W1 = np.random.randn(input_size, hidden_size) * limit1
        self.b1 = np.zeros((1, hidden_size))
        
        # Xavier initialization for hidden-to-output weights
        limit2 = np.sqrt(2.0 / (hidden_size + output_size))
        self.W2 = np.random.randn(hidden_size, output_size) * limit2
        self.b2 = np.zeros((1, output_size))
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
    
    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function: σ(z) = 1 / (1 + exp(-z))
        Clipped to prevent overflow in exp.
        
        Args:
            z (np.ndarray): Pre-activation values.
        
        Returns:
            np.ndarray: Activated values in range (0, 1).
        """
        z_clipped = np.clip(z, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    @staticmethod
    def sigmoid_derivative(a):
        """
        Derivative of sigmoid function: σ'(z) = σ(z) * (1 - σ(z)) = a * (1 - a)
        where a = σ(z) is the already-activated value.
        
        Args:
            a (np.ndarray): Sigmoid-activated values.
        
        Returns:
            np.ndarray: Derivative values.
        """
        return a * (1.0 - a)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Computes: input → hidden (sigmoid) → output (sigmoid)
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_size).
        
        Returns:
            tuple: (hidden_output, final_output)
                - hidden_output: shape (n_samples, hidden_size)
                - final_output: shape (n_samples, output_size)
        """
        # Hidden layer: z1 = X @ W1 + b1, a1 = sigmoid(z1)
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer: z2 = a1 @ W2 + b2, a2 = sigmoid(z2)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a1, self.a2
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute Mean Squared Error loss.
        
        MSE = (1/n) * Σ (y_true - y_pred)²
        
        Args:
            y_true (np.ndarray): True labels, shape (n_samples, output_size).
            y_pred (np.ndarray): Predicted outputs, shape (n_samples, output_size).
        
        Returns:
            float: Mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y_true, learning_rate=0.1):
        """
        Backward pass (backpropagation) to update weights.
        
        Computes gradients of MSE loss w.r.t. all weights and biases,
        then updates them using gradient descent.
        
        Math:
            δ2 = (a2 - y) * σ'(z2)           [output layer error]
            δ1 = (δ2 @ W2.T) * σ'(z1)        [hidden layer error]
            ∂L/∂W2 = a1.T @ δ2               [hidden-to-output gradient]
            ∂L/∂W1 = X.T @ δ1                [input-to-hidden gradient]
        
        Args:
            X (np.ndarray): Input data, shape (n_samples, input_size).
            y_true (np.ndarray): True labels, shape (n_samples, output_size).
            learning_rate (float): Step size for weight updates.
        """
        n_samples = X.shape[0]
        
        # Output layer error: δ2 = (predicted - actual) * sigmoid_derivative(output)
        output_error = self.a2 - y_true
        output_delta = output_error * self.sigmoid_derivative(self.a2)
        
        # Hidden layer error: δ1 = (δ2 @ W2.T) * sigmoid_derivative(hidden)
        hidden_error = output_delta @ self.W2.T
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Compute gradients (averaged over batch)
        dW2 = (self.a1.T @ output_delta) / n_samples
        db2 = np.sum(output_delta, axis=0, keepdims=True) / n_samples
        dW1 = (X.T @ hidden_delta) / n_samples
        db1 = np.sum(hidden_delta, axis=0, keepdims=True) / n_samples
        
        # Update weights and biases using gradient descent
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000, learning_rate=0.5, verbose=True, print_every=100):
        """
        Train the neural network using backpropagation.
        
        Performs full-batch gradient descent for the specified number of epochs.
        Records loss and accuracy at each epoch for later plotting.
        
        Args:
            X (np.ndarray): Training input, shape (n_samples, input_size).
            y (np.ndarray): Training labels (one-hot), shape (n_samples, output_size).
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate for gradient descent.
            verbose (bool): Whether to print progress.
            print_every (int): Print frequency (every N epochs).
        
        Returns:
            tuple: (loss_history, accuracy_history) — lists of loss and accuracy per epoch.
        """
        self.loss_history = []
        self.accuracy_history = []
        
        for epoch in range(epochs):
            # Forward pass
            _, output = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)
            
            # Compute accuracy
            predictions = np.argmax(output, axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels) * 100
            self.accuracy_history.append(accuracy)
            
            # Backward pass (update weights)
            self.backward(X, y, learning_rate)
            
            # Print progress
            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:5d}/{epochs} | Loss: {loss:.6f} | Accuracy: {accuracy:.1f}%")
        
        return self.loss_history, self.accuracy_history
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X (np.ndarray): Input data, shape (n_samples, input_size).
        
        Returns:
            tuple: (predicted_classes, probabilities)
                - predicted_classes: np.ndarray of class indices
                - probabilities: np.ndarray of output activations
        """
        _, output = self.forward(X)
        predicted_classes = np.argmax(output, axis=1)
        return predicted_classes, output
    
    def get_accuracy(self, X, y):
        """
        Compute classification accuracy on a dataset.
        
        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): One-hot encoded labels.
        
        Returns:
            float: Accuracy as a percentage.
        """
        predictions, _ = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels) * 100
    
    def save(self, filepath):
        """
        Save model weights and biases to a .npz file.
        
        Args:
            filepath (str): Path to save the model.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 input_size=self.input_size,
                 hidden_size=self.hidden_size,
                 output_size=self.output_size)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model weights and biases from a .npz file.
        
        Args:
            filepath (str): Path to the saved model.
        """
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.input_size = int(data['input_size'])
        self.hidden_size = int(data['hidden_size'])
        self.output_size = int(data['output_size'])
        print(f"Model loaded from {filepath} (architecture: {self.input_size}-{self.hidden_size}-{self.output_size})")
