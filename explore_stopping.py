import random
import matplotlib.pyplot as plt

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
                solution_found_count[str(i)] += 1
                if candidate == optimal_candidate:
                    optimal_solution_found_count[str(i)] += 1
                
                break



print(candidates)
print(optimal_candidate)
x, y = zip(*optimal_solution_found_count.items())

print(x)
print(y)
print(optimal_solution_found_count)
plt.plot(x,y)
plt.show()

