# Optimal-Stopping

## Team Members
Michael Grajera
Benson Shafer

## Required Packages

Install the required Python libraries using `pip`. Run the following command to install all dependencies:

* matplotlib
```bash
pip install matplotlib
```
* numpy
```bash
pip install numpy
```

## Running the File

In the shell run
```bash
py HW1_Shafer_Benson.py {args}
```

These are the different arguments and their output

```bash
py HW1_Shafer_Benson.py 1 true
```
Graph for part 1 (Omit true to run 10 iterations of the experiment and find the average)

```bash
py HW1_Shafer_Benson.py 2
```
Graphs for part 2


## Notes and Documentation
Part 2: Exploring Alternative Distributions

**Analyze how the optimal stopping threshold changes compared to the uniform distribution case. Adjust your algorithm accordingly.**

In the uniform distribution, you’re equally likely to get any number from 1 to 99, so waiting longer (a higher threshold) maximizes reward potential.
In the normal distribution, you don’t need to wait as long to encounter values close to the mean. So, the optimal threshold will be lower (around 50).
 

**Further, explore a scenario where the data follows a skewed beta distribution, specifically Beta(2,7). Analyze the impact of this skew on your stopping strategy.**

In a Beta(2,7) distribution, the data is skewed towards lower values, so waiting for high values isn't as useful because they’re much less frequent. The optimal strategy will involve stopping earlier (with a lower threshold), allowing you to maximize the potential reward before you invest too much time.
For example, if the optimal stopping threshold for the normal distribution is around 50, you might find that it’s closer to 30-40 for the Beta distribution, because waiting for high values in this skewed distribution is less rewarding.
 
Optimal Stopping Threshold - Uniform: 10
Optimal Stopping Threshold - Normal: 5
Optimal Stopping Threshold - Beta: 9

**Run sufficient simulations for both distributions to ascertain the optimal stopping thresholds and compare these results with the uniform distribution scenario.**

Uniform Distribution: Because every value has an equal chance of appearing, you can afford to set a higher stopping threshold and wait for higher values. The optimal threshold may be around 70.
Normal Distribution: Since values cluster around 50, the stopping threshold should be set lower, likely around 50, since waiting too long doesn’t provide additional benefits.
Beta(2,7) Distribution: The skew means higher values are rare, so you should select earlier (with a threshold around 30-40). Waiting for high values doesn’t pay off because they appear much less frequently.
Key patterns:
Distributions with more spread-out or uniform values favor a later stopping threshold.
Concentrated (normal) or skewed (beta) distributions favor an earlier stopping threshold.
 

**Discuss any patterns or insights you observe regarding how distribution shape affects the stopping point.**

Uniform Distribution: With no preference for certain values, the strategy is to be more selective, hence a higher threshold.
Normal Distribution: You can stop earlier because the data clusters around a central value (mean of 50).
Skewed Beta Distribution: With skewed distributions, it’s optimal to stop earlier because high values are less likely to appear.
