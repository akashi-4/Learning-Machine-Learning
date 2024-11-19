import pandas as pd
from scipy.stats import shapiro
import seaborn as sns
import matplotlib.pyplot as plt

def get_max_accuracy():
    try:
        # Read the CSV file
        df = pd.read_csv('results/accuracy.csv')
        
        # Group by model_name and get the maximum accuracy for each
        max_accuracy = df.groupby('algorithm')['accuracy'].max().reset_index()
        
        # Sort by accuracy in descending order
        max_accuracy = max_accuracy.sort_values('accuracy', ascending=False)
        
        # Round accuracy to 4 decimal places for cleaner display
        max_accuracy['accuracy'] = max_accuracy['accuracy'].round(5)
        
        return max_accuracy
    except FileNotFoundError:
        print("accuracy.csv file not found in results directory")
        return None

#print(get_max_accuracy())

resultados = [0.988, 0.986, 0.982,0.982,0.968,0.968,0.948,0.946,0.938,0.838, 0.826]

shapiro(resultados)

sns.kdeplot(resultados)
#plt.show()

from scipy.stats import f_oneway

# Example data
data1 = [0.988, 0.986, 0.982,0.982,0.968,0.968,0.948,0.946,0.938,0.838, 0.826]
data2 = [0.988, 0.986, 0.982,0.982,0.968,0.968,0.948,0.946,0.938,0.838, 0.826]
data3 = [0.988, 0.986, 0.982,0.982,0.968,0.968,0.948,0.946,0.938,0.838, 0.826]

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def randomize_accuracies(base_data, noise_level=0.02):
    """
    Add random noise to accuracy values while keeping them in valid range (0-1)
    """
    noise = np.random.uniform(-noise_level, noise_level, len(base_data))
    randomized = np.clip(np.array(base_data) + noise, 0, 1)
    return sorted(randomized, reverse=True)

# Randomize with different noise levels for each algorithm
algorithm1_accuracies = randomize_accuracies(data1, noise_level=0.015)
algorithm2_accuracies = randomize_accuracies(data2, noise_level=0.02)
algorithm3_accuracies = randomize_accuracies(data3, noise_level=0.025)

#print("Algorithm 1 accuracies:", [round(x, 3) for x in algorithm1_accuracies])
#print("Algorithm 2 accuracies:", [round(x, 3) for x in algorithm2_accuracies])
#print("Algorithm 3 accuracies:", [round(x, 3) for x in algorithm3_accuracies])

# Create a dictionary for easy access
algorithm_accuracies = {
    'Decision Tree': algorithm1_accuracies,
    'Random Forest': algorithm2_accuracies,
    'KNN': algorithm3_accuracies
}

_, p = f_oneway(algorithm_accuracies['Decision Tree'], algorithm_accuracies['Random Forest'], algorithm_accuracies['KNN'])

alpha = 0.05
if p<= alpha:
    print('Hipótese nula rejeitada. Dados são diferentes')
else:
    print('Hipótese nula aceita. Dados são iguais') 

from statsmodels.stats.multicomp import MultiComparison

# Create a list with the algorithm names
algorithm_names = ['Decision Tree'] * len(algorithm1_accuracies) + ['Random Forest'] * len(algorithm2_accuracies) + ['KNN'] * len(algorithm3_accuracies)

# Combine all accuracies into a single list
all_accuracies = algorithm1_accuracies + algorithm2_accuracies + algorithm3_accuracies

# Create MultiComparison object for ANOVA analysis
mc = MultiComparison(all_accuracies, algorithm_names)

# Print the pairwise comparisons
result = mc.tukeyhsd()
print(result)

# Plot the results
result.plot_simultaneous()
plt.show()

# How do we interpret the results?
# If the confidence interval contains zero, the difference is not significant
# If the confidence interval does not contain zero, the difference is significant
# The wider the confidence interval, the less significant the difference
# The narrower the confidence interval, the more significant