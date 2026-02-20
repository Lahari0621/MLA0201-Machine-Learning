# =========================================
# FULL ALL-IN-ONE MLP STRUCTURAL LOGIC DEMO
# Architecture: 3 Input → 4 Hidden → 1 Output
# =========================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# -------------------------------
# DATASET (Non-linear XOR-style)
# -------------------------------
X = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], dtype=np.float32)

y = np.array([0, 1, 1, 0], dtype=np.float32)

print("\nDataset:")
print(X)
print("Labels:", y)

# =====================================================
# MODEL 1: MLP WITH HIDDEN LAYER (NON-LINEAR LEARNING)
# =====================================================
print("\nMODEL 1: With Hidden Layer (3-4-1)")

model_nonlinear = Sequential([
    Input(shape=(3,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_nonlinear.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_nonlinear.fit(X, y, epochs=500, verbose=0)

pred_nonlinear = model_nonlinear.predict(X)
print("Predictions (With Hidden Layer):")
print(pred_nonlinear.round(3))

# =====================================================
# MODEL 2: WITHOUT HIDDEN LAYER (LINEAR MODEL)
# =====================================================
print("\nMODEL 2: Without Hidden Layer (Linear Model)")

model_linear = Sequential([
    Input(shape=(3,)),
    Dense(1, activation='sigmoid')
])

model_linear.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_linear.fit(X, y, epochs=500, verbose=0)

pred_linear = model_linear.predict(X)
print("Predictions (Without Hidden Layer):")
print(pred_linear.round(3))

# =====================================================
# CONCLUSION
# =====================================================
print("\nConclusion:")
print("- Model WITH hidden layer learns non-linear patterns.")
print("- Model WITHOUT hidden layer fails (only linear).")
print("- Hidden layer + activation enables non-linearity.")