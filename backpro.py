import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# Initialize weights
np.random.seed(0)
W1 = np.random.randn(2,6) * 0.1  # hidden layer size 6
b1 = np.zeros((1,6))
W2 = np.random.randn(6,1) * 0.1
b2 = np.zeros((1,1))

# Learning rate and epochs
lr = 0.05
epochs = 100  # reduced epochs

loss_history = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # Compute loss (MSE)
    loss = np.mean((a2 - Y)**2)
    loss_history.append(loss)
    
    # Backpropagation
    d_a2 = 2*(a2 - Y)/Y.size
    d_z2 = d_a2 * sigmoid_derivative(z2)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(z1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    
    # Update weights
    W1 -= lr * d_W1
    b1 -= lr * d_b1
    W2 -= lr * d_W2
    b2 -= lr * d_b2

# Plot the loss curve
plt.plot(loss_history, color='blue')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve (XOR Backpropagation, 100 Epochs)")
plt.grid(True)
plt.show()

print(f"Final loss after {epochs} epochs: {loss_history[-1]:.6f}")
