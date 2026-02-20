import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


# ===============================
# 1️⃣ DEEP LINEAR NETWORK
# ===============================
print("1️⃣ Deep Linear Network (All Linear Activations)")

x = np.array([1, 0])

W1 = np.random.randn(2, 3)
W2 = np.random.randn(3, 1)

y_linear = x @ W1 @ W2

print("Output:", y_linear)
print("➡ Multiple linear layers collapse into ONE linear layer\n")


# ===============================
# 2️⃣ LINEAR NETWORK ON XOR
# ===============================
print("2️⃣ Linear Network trying to solve XOR")

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

W = np.random.randn(2, 1)
b = np.random.randn(1)

y_xor_linear = X @ W + b

print("Linear Outputs:")
print(y_xor_linear)
print("➡ XOR is NOT linearly separable → FAIL\n")


# ===============================
# 3️⃣ NON-LINEAR NETWORK (SIGMOID)
# ===============================
print("3️⃣ Non-Linear Network with SIGMOID")

W1_sig = np.random.randn(2, 3)
b1_sig = np.random.randn(3)

W2_sig = np.random.randn(3, 1)
b2_sig = np.random.randn(1)

hidden_sig = sigmoid(X @ W1_sig + b1_sig)
output_sig = sigmoid(hidden_sig @ W2_sig + b2_sig)

print("Sigmoid Outputs:")
print(output_sig)
print("➡ Sigmoid introduces non-linearity\n")


# ===============================
# 4️⃣ NON-LINEAR NETWORK (RELU)
# ===============================
print("4️⃣ Non-Linear Network with ReLU")

W1_relu = np.random.randn(2, 3)
b1_relu = np.random.randn(3)

W2_relu = np.random.randn(3, 1)
b2_relu = np.random.randn(1)

hidden_relu = relu(X @ W1_relu + b1_relu)
output_relu = hidden_relu @ W2_relu + b2_relu

print("ReLU Outputs:")
print(output_relu)
print("➡ ReLU introduces non-linearity\n")


# ===============================
# FINAL SUMMARY
# ===============================
print("SUMMARY:")
print("• Linear layers only → behave as one linear model")
print("• Linear networks → cannot solve XOR")
print("• Sigmoid & ReLU → enable non-linear learning")
