import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Assuming you have a DataFrame df with your data
df = pd.read_csv('your_data.csv')  # replace 'your_data.csv' with your actual data file

# Specify your independent variable
independent_var = 'anti-social behavior'

# Specify your dependent variables
dependent_vars = ['Physical Aggression', 'Social Aggression', 'Rule-Breaking']

# Create a new column that is the sum of the dependent variables
df['combined'] = df[dependent_vars].sum(axis=1)

# Calculate correlation
correlation = df[[independent_var, 'combined']].corr()

# Display correlation
print("Correlation:\n", correlation)

# Plot correlation
sns.heatmap(correlation, annot=True, cmap='coolwarm')

# Plot linear regression for the combined dependent variable
sns.regplot(x=independent_var, y='combined', data=df)
plt.savefig('regression_plot_combined.png')  # saves the plot as an image file
plt.show()

# Calculate descriptive statistics
desc_stats = df.describe()

# Calculate variance
variance = df.var()

# Display descriptive statistics and variance
print("\nDescriptive Statistics:\n", desc_stats)
print("\nVariance:\n", variance)

# Perform one-way ANOVA test
f_val, p_val = f_oneway(df['Physical Aggression'], df['Social Aggression'], df['Rule-Breaking'])

# Display F-value and p-value
print("\nOne-way ANOVA")
print("F =", f_val)
print("p =", p_val)