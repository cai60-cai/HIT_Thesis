import pandas as pd
import numpy as np

# Create the data
data = {
    'Dimension': [10, 20, 30, 40, 50, 60, 70, 78],
    'Var Explained': [0.9479, 0.9966, 0.9993, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000],
    'Accuracy': [0.9676, 0.9871, 0.9834, 0.9840, 0.9861, 0.9853, 0.9972, 0.9914],
    'F1': [0.9682, 0.9872, 0.9834, 0.9842, 0.9862, 0.9852, 0.9972, 0.9914],
    'Model Size': [427665] * 8,
    'Inference Time': [7.5043, 22.7118, 22.7077, 22.7028, 22.6541, 22.6585, 23.1571, 23.0785]
}

df = pd.DataFrame(data)

# Calculate dimension reduction percentage
df['Dim Reduction'] = (78 - df['Dimension']) / 78

# Min-max normalize inference time
min_time = df['Inference Time'].min()
max_time = df['Inference Time'].max()
time_normalized = 1 - (df['Inference Time'] - min_time) / (max_time - min_time)

# Calculate score
df['Score'] = (0.3 * df['Accuracy'] + 
               0.3 * df['F1'] + 
               0.1 * time_normalized +
               0.2 * df['Dim Reduction'])

# Round all numeric columns to 4 decimal places
for col in df.columns:
    if col != 'Dimension' and col != 'Model Size':
        df[col] = df[col].round(4)

# Display results
print("\nResults Table:")
print(df.to_string(index=False))

# Find best score
best_row = df.loc[df['Score'].idxmax()]
print(f"\nBest performing dimension: {int(best_row['Dimension'])} with score: {best_row['Score']:.4f}")

import matplotlib.pyplot as plt

# Plot score vs Dimension
plt.figure(figsize=(8, 6))
plt.plot(df['Dimension'], df['Score'], marker='o', linestyle='-', color='b')

# Annotate the points with the corresponding scores
for i, row in df.iterrows():
    plt.text(row['Dimension'], row['Score'], f"{row['Score']:.4f}", ha='center', va='bottom', fontsize=9)

# Add titles and labels
plt.title('Score vs Dimension', fontsize=14)
plt.xlabel('Dimension', fontsize=12)
plt.ylabel('Score', fontsize=12)

# Show the plot
plt.grid(True)
plt.savefig("score.png")