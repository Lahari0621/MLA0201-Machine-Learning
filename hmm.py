# Hidden Markov Model (HMM) - Weather Prediction
# States: Rainy, Sunny
# Observations: Walk, Shop, Clean

import numpy as np
from hmmlearn import hmm

# -------------------------------
# Step 1: Define the HMM Model
# -------------------------------
model = hmm.MultinomialHMM(
    n_components=2,
    n_iter=100,
    random_state=42
)

# -------------------------------
# Step 2: Initialize Model Parameters
# -------------------------------

# Initial state probabilities
# P(Rainy) = 0.6, P(Sunny) = 0.4
model.startprob_ = np.array([0.6, 0.4])

# State transition probabilities
# Rainy -> Rainy/Sunny
# Sunny -> Rainy/Sunny
model.transmat_ = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Emission probabilities
# Observations: Walk(0), Shop(1), Clean(2)
model.emissionprob_ = np.array([
    [0.1, 0.4, 0.5],  # Rainy
    [0.6, 0.3, 0.1]   # Sunny
])

# -------------------------------
# Step 3: Observation Sequence
# -------------------------------
# Walk -> Shop -> Clean
observations = np.array([[0], [1], [2]])

# -------------------------------
# Step 4: Forward Algorithm
# Compute Sequence Probability
# -------------------------------
log_probability = model.score(observations)
sequence_probability = np.exp(log_probability)

print("=== Forward Algorithm ===")
print("Log Probability of Observation Sequence:", log_probability)
print("Sequence Probability:", sequence_probability)

# -------------------------------
# Step 5: Viterbi Algorithm
# Decode Most Likely State Sequence
# -------------------------------
log_prob, hidden_states = model.decode(
    observations,
    algorithm="viterbi"
)

state_map = {0: "Rainy", 1: "Sunny"}
decoded_states = [state_map[state] for state in hidden_states]

print("\n=== Viterbi Algorithm ===")
print("Log Probability of Best State Sequence:", log_prob)
print("Most Likely Weather Sequence:")
print(decoded_states)

# -------------------------------
# Step 6: Display Observation Mapping
# -------------------------------
obs_map = {0: "Walk", 1: "Shop", 2: "Clean"}
decoded_observations = [obs_map[int(obs[0])] for obs in observations]

print("\nObservation Sequence:")
print(decoded_observations)
