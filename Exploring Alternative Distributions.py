import numpy as np
import matplotlib.pyplot as plt

# Function to simulate and calculate average reward for a given stopping threshold
def simulate_distribution(simulations, distribution_func, max_evals=100):
    avg_rewards = []
    for stopping_threshold in range(1, max_evals):
        total_reward = 0
        for _ in range(simulations):
            best_value = -1
            evaluations = 0
            for i in range(max_evals):
                value = distribution_func()
                evaluations += 1
                if value > best_value:
                    best_value = value
                if evaluations >= stopping_threshold:
                    break
            total_reward += best_value - evaluations
        avg_rewards.append(total_reward / simulations)
    return avg_rewards

# Distribution functions
def uniform_distribution():
    return np.random.randint(1, 100)

def normal_distribution():
    return min(99, max(1, int(np.random.normal(50, 10))))

def beta_distribution():
    return min(99, max(1, int(np.random.beta(2, 7) * 99)))

# Run simulations
simulations = 1000
uniform_rewards = simulate_distribution(simulations, uniform_distribution)
normal_rewards = simulate_distribution(simulations, normal_distribution)
beta_rewards = simulate_distribution(simulations, beta_distribution)

# Find the optimal stopping threshold (index of the maximum reward)
optimal_uniform = np.argmax(uniform_rewards) + 1
optimal_normal = np.argmax(normal_rewards) + 1
optimal_beta = np.argmax(beta_rewards) + 1

# Generate sample data for histograms
uniform_samples = [uniform_distribution() for _ in range(10000)]
normal_samples = [normal_distribution() for _ in range(10000)]
beta_samples = [beta_distribution() for _ in range(10000)]

# Plot all the graphs in subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Reward vs Stopping Threshold for all distributions
thresholds = range(1, len(uniform_rewards) + 1)
axs[0, 0].plot(thresholds, uniform_rewards, label='Uniform Distribution', color='blue')
axs[0, 0].plot(thresholds, normal_rewards, label='Normal Distribution (mean=50, sd=10)', color='green')
axs[0, 0].plot(thresholds, beta_rewards, label='Beta Distribution (2, 7)', color='red')
axs[0, 0].set_xlabel('Stopping Threshold')
axs[0, 0].set_ylabel('Average Reward')
axs[0, 0].set_title('Reward vs Stopping Threshold')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Subplot 2: Overlay of Optimal Stopping Thresholds
axs[0, 1].plot(thresholds, uniform_rewards, label='Uniform Distribution', color='blue')
axs[0, 1].plot(thresholds, normal_rewards, label='Normal Distribution (mean=50, sd=10)', color='green')
axs[0, 1].plot(thresholds, beta_rewards, label='Beta Distribution (2, 7)', color='red')
axs[0, 1].axvline(optimal_uniform, color='blue', linestyle='--', label=f'Optimal Uniform ({optimal_uniform})')
axs[0, 1].axvline(optimal_normal, color='green', linestyle='--', label=f'Optimal Normal ({optimal_normal})')
axs[0, 1].axvline(optimal_beta, color='red', linestyle='--', label=f'Optimal Beta ({optimal_beta})')
axs[0, 1].set_xlabel('Stopping Threshold')
axs[0, 1].set_ylabel('Average Reward')
axs[0, 1].set_title('Optimal Stopping Thresholds')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Subplot 3: Histogram of Uniform Distribution
axs[1, 0].hist(uniform_samples, bins=30, alpha=0.6, color='blue', density=True)
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title('Histogram of Uniform Distribution')
axs[1, 0].grid(True)

# Subplot 4: Histograms of Normal and Beta Distributions
axs[1, 1].hist(normal_samples, bins=30, alpha=0.6, color='green', label='Normal Distribution (mean=50, sd=10)', density=True)
axs[1, 1].hist(beta_samples, bins=30, alpha=0.6, color='red', label='Beta Distribution (2, 7)', density=True)
axs[1, 1].set_xlabel('Value')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title('Histograms of Normal and Beta Distributions')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
