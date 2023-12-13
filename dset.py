import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame df with your data
df = pd.read_csv('your_data.csv')  # replace 'your_data.csv' with your actual data file

# Specify your independent variable
independent_var = 'independent_var'  # replace 'independent_var' with your actual independent variable

# Specify your dependent variables
dependent_vars = ['dependent_var1', 'dependent_var2']  # replace these with your actual dependent variables

# Calculate correlation
correlation = df[[independent_var] + dependent_vars].corr()

# Display correlation
print(correlation)

# Plot correlation
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()