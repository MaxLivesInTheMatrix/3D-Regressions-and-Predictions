# First We'll make some general observations about the data
# Think simple scatter plots of where students feel they are at
# Then we want to find out other stats, particulary the PCA of GPA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

filePath = r"/Users/max/gitRepositories/3D-Regressions-and-Predictions/student_lifestyle_dataset.csv"

df = pd.read_csv(filePath)

# Drop non-numeric columns if necessary
numeric_df = df.drop(columns=['Student_ID'], errors='ignore')
print(df)

def plotDistributionSimp(columnName):
    # Create figure with gridspec for custom subplot layouts
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    
    # First subplot for distribution
    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()
    
    # Plot density on the left axis (ax1)
    sns.kdeplot(data=df[columnName], color='red', linewidth=2, label='Density Plot', ax=ax1)
    
    # Plot histogram on the right axis (ax2)
    counts, bins, _ = ax2.hist(df[columnName], bins=20, color='skyblue', 
                              alpha=0.6, edgecolor='black', label='Histogram')
    
    # Customize left axis (Density)
    ax1.set_ylabel('Density', fontsize=14, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    
    # Customize right axis (Frequency)
    ax2.set_ylabel('Frequency', fontsize=14, color='skyblue')
    ax2.tick_params(axis='y', labelcolor='skyblue')
    
    # Add title and x-label
    ax1.set_title(f'{columnName} Distribution: Histogram + Density Plot', fontsize=16)
    ax1.set_xlabel(f'{columnName}', fontsize=14)
    
    # Add grid (only for left axis)
    ax1.grid(axis='y', alpha=0.75)
    
    # Add legends for both plots
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
    
    # Second subplot for descriptive statistics
    ax3 = fig.add_subplot(gs[1])
    ax3.axis('tight')
    ax3.axis('off')
    
    # Get descriptive statistics
    stats = df[columnName].describe()
    
    # Create table with statistics
    table_data = [[round(stats[i], 3)] for i in stats.index]
    table = ax3.table(cellText=table_data,
                     rowLabels=stats.index,
                     colLabels=[f'{columnName} Statistics'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.2, 0, 0.6, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.show()

    
gpaColumn = 'GPA'
studyHours = 'Study_Hours_Per_Day'
extracurricularHours = 'Extracurricular_Hours_Per_Day'
sleepHours = 'Sleep_Hours_Per_Day'
socialHours = 'Social_Hours_Per_Day'
physicalActivity = 'Physical_Activity_Hours_Per_Day'
stressLevel = "Stress_Level"

plotDistributionSimp(gpaColumn)
plotDistributionSimp(studyHours)
plotDistributionSimp(extracurricularHours)
plotDistributionSimp(sleepHours)
plotDistributionSimp(socialHours)
plotDistributionSimp(physicalActivity)
plotDistributionSimp(stressLevel)



# # Calculate statistics
# statistics = {
#     'Max': numeric_df.max(),
#     'Min': numeric_df.min(),
#     'Median': numeric_df.median(),
#     'Mean': numeric_df.mean(),
#     'Standard Deviation': numeric_df.std(),
#     'Variance': numeric_df.var()
# }

# # Convert statistics to a DataFrame for better readability
# stats_df = pd.DataFrame(statistics)

# # Print the result
# print(stats_df)

# # Standardize the data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(numeric_df)

# # Perform PCA
# pca = PCA()  # Create PCA instance (default: all components)
# pca_result = pca.fit_transform(scaled_data)

# # Create a DataFrame for PCA results
# pca_df = pd.DataFrame(
#     pca_result,
#     columns=[f"PC{i+1}" for i in range(pca.n_components_)]
# )

# # Add explained variance ratio for each component
# explained_variance = pca.explained_variance_ratio_

# # Display results
# print("PCA Result:")
# print(pca_df.head())

# print("\nExplained Variance Ratio:")
# for i, ratio in enumerate(explained_variance, start=1):
#     print(f"PC{i}: {ratio:.2f}")
    

# # Scatter plot for PC1 and PC2
# plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', edgecolor='k')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA - First Two Components')
# plt.show()