import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame with dummy data
data = {
    'independent_var': [1, 2, 3, 4, 5],
    'dependent_var1': [2, 4, 6, 8, 10],
    'dependent_var2': [1, 3, 5, 7, 9]
}
df = pd.DataFrame(data)

# Specify your independent variable
independent_var = 'independent_var'

# Specify your dependent variables
dependent_vars = ['dependent_var1', 'dependent_var2']

# Calculate correlation
correlation = df[[independent_var] + dependent_vars].corr()

# Display correlation
print(correlation)

# Plot correlation
sns.heatmap(correlation, annot=True, cmap='coolwarm')

# Plot linear regression for each dependent variable
for dependent_var in dependent_vars:
    sns.regplot(x=independent_var, y=dependent_var, data=df)
    plt.show()
    
    # Plot linear regression for each dependent variable
for i, dependent_var in enumerate(dependent_vars):
    sns.regplot(x=independent_var, y=dependent_var, data=df)
    plt.savefig(f'regression_plot_{i}.png')  # saves the plot as an image file
    plt.show()