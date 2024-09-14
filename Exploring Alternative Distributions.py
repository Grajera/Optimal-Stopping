import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
num_simulations = 10000  # Number of simulations to run
max_evaluations = 100  # Maximum number of evaluations

# Function to simulate rewards for a given stopping threshold and distribution
def simulate_rewards(distribution, threshold, max_evaluations, num_simulations):
    rewards = []
    for _ in range(num_simulations):
        evaluations = 0
        best_value = -np.inf
        for i in range(max_evaluations):
            # Get the next value from the distribution
            if distribution == "uniform":
                value = np.random.randint(1, 100)  # Uniform distribution from 1 to 99
            elif distribution == "normal":
                value = min(99, max(1, int(np.random.normal(50, 10))))  # Normal distribution
            elif distribution == "beta":
                value = min(99, max(1, int(np.random.beta(2, 7) * 99)))  # Beta distribution scaled to 99
            else:
                raise ValueError("Invalid distribution type")

            evaluations += 1
            # If the value exceeds the threshold, select it and stop
            if value > threshold:
                best_value = value
                break

        # Calculate the reward
        reward = best_value - evaluations
        rewards.append(reward)

    return np.mean(rewards)

# Simulate the optimal stopping threshold for uniform, normal, and beta distributions
thresholds = range(1, 100)
uniform_rewards = []
normal_rewards = []
beta_rewards = []

for threshold in thresholds:
    uniform_rewards.append(simulate_rewards("uniform", threshold, max_evaluations, num_simulations))
    normal_rewards.append(simulate_rewards("normal", threshold, max_evaluations, num_simulations))
    beta_rewards.append(simulate_rewards("beta", threshold, max_evaluations, num_simulations))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(thresholds, uniform_rewards, label="Uniform Distribution", color="blue")
plt.plot(thresholds, normal_rewards, label="Normal Distribution", color="red")
plt.plot(thresholds, beta_rewards, label="Beta(2,7) Distribution", color="green")
plt.xlabel("Stopping Threshold")
plt.ylabel("Average Reward")
plt.title("Average Reward vs Stopping Threshold (Uniform, Normal, Beta Distributions)")
plt.legend()
plt.grid(True)
plt.show()
