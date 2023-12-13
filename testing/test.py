import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, pearsonr
from scipy import stats

# Set a seed for reproducibility
np.random.seed(0)

# Create a DataFrame with random data for 50 respondents
data = {
    'anti-social behavior': np.random.randint(1, 10, 50),
    'Physical Aggression': np.random.randint(1, 10, 50),
    'Social Aggression': np.random.randint(1, 10, 50),
    'Rule-Breaking': np.random.randint(1, 10, 50)
}
df = pd.DataFrame(data)

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
plt.savefig('correlation_heatmap.png')  # saves the heatmap as an image file
plt.show()

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

# Calculate Pearson correlation coefficient and the p-value
corr_coeff, p_value = pearsonr(df[independent_var], df['combined'])

# Display Pearson correlation coefficient and p-value
print("\nPearson Correlation Coefficient: ", corr_coeff)
print("p-value: ", p_value)

# Conclusion
if p_value < 0.05:
    print("The correlation between the independent and dependent variables is statistically significant.")
else:
    print("The correlation between the independent and dependent variables is not statistically significant.")

# Additional statistical tools

# Pairplot
sns.pairplot(df)
plt.savefig('pairplot.png')  # saves the pairplot as an image file
plt.show()

# Box plots for each variable
for var in [independent_var] + dependent_vars:
    sns.boxplot(x=df[var])
    plt.savefig(f'boxplot_{var}.png')  # saves each boxplot as an image file
    plt.show()

# Confidence Intervals
confidence_intervals = stats.norm.interval(0.95, loc=df.mean(), scale=df.std())
print("\nConfidence Intervals:\n", confidence_intervals)

# Conclusion for Confidence Intervals
print("\nConclusion for Confidence Intervals:")
print("The confidence intervals provide a range within which the true population parameter lies with a confidence level of 95%. If the confidence interval for a parameter includes zero, it suggests that there's no effect.")

# Covariance Matrix
cov_matrix = df.cov()
print("\nCovariance Matrix:\n", cov_matrix)

# Conclusion for Covariance Matrix
print("\nConclusion for Covariance Matrix:")
print("The covariance matrix provides a measure of how much each of the variables change together. If the covariance is positive, the variables increase together; if it's negative, one variable decreases as the other increases.")

# Conclusion for Pearson Correlation
print("\nConclusion for Pearson Correlation:")
if p_value < 0.05:
    print("The correlation between the independent and dependent variables is statistically significant.")
else:
    print("The correlation between the independent and dependent variables is not statistically significant.")