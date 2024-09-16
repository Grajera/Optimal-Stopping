import random
import numpy as np
import matplotlib.pyplot as plt
import sys

    
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

def general_optimum(graph):
    len_candidates = 100
    solution_found_count = {}
    optimal_solution_found_count = {}
    for i in range(1, len_candidates):
        solution_found_count[str(i)] = 0
        optimal_solution_found_count[str(i)] = 0
    
    for experiment in range(1000):
        candidates = random.sample(range(0,1000), len_candidates)
        optimal_candidate = max(candidates)

        for i in range(1, len_candidates):
            for candidate in candidates[i:-1]:
                if candidate > max(candidates[0:i]):
                    if candidate == optimal_candidate:
                        optimal_solution_found_count[str(i)] += 1
                    break

    if (graph):
        x, y = zip(*optimal_solution_found_count.items())

        plt.figure(figsize=(10, 6))
        plt.xticks(ticks=range(0, 101, 5))
        plt.plot(x,y)
        plt.scatter(max(optimal_solution_found_count, key=optimal_solution_found_count.get), max(optimal_solution_found_count.values()), color='red', label="Optimal Stopping: " + max(optimal_solution_found_count, key=optimal_solution_found_count.get) + "%")
        plt.ylabel("Optimal Solutions Found")
        plt.xlabel("Stopping Point")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        return max(optimal_solution_found_count, key=optimal_solution_found_count.get)
    
def exploring_alternative_distribution():
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


def main():
    experiment_count = 10
    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            if len(sys.argv) > 2 and sys.argv[2].lower() == "true":
                general_optimum(True)
            else:
                percentage_count = []
                for i in range(experiment_count):
                    percentage_count.append(general_optimum(False))
                np_array = np.array(percentage_count)
                print("The average percentage over " + str(experiment_count) + " iterations is:")
                print(np.mean(np_array.astype(int)))

        if sys.argv[1] == "2":
            exploring_alternative_distribution()
    else:
        print("Input either a 1, 2, 3 as the argument for the different test results")



if __name__ == '__main__':
    main()