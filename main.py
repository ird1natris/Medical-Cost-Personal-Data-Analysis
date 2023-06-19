import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, zscore

# Load the dataset
df = pd.read_csv('insurance.csv')

# Exploratory Data Analysis

# Check the first few rows of the dataset
print(df.head())

# Get information about the dataset
print(df.info())

# Descriptive statistics of numerical columns
print(df.describe())

# Count the number of missing values in each column
print(df.isnull().sum())

# Identify unique values in categorical columns
categorical_cols = ['sex', 'smoker', 'region']
for col in categorical_cols:
    unique_values = df[col].unique()
    print(f'Unique values in {col}: {unique_values}')

# Data Preprocessing

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

# Split the data into features and target variable
X = df.drop('charges', axis=1)
y = df['charges']

# Data Visualization

# Plot the distribution of charges
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='charges', kde=True)
plt.title('Distribution of Charges')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot of age vs. charges
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='age', y='charges', hue='smoker_yes')
plt.title('Age vs. Charges')
plt.show()

# Box plot of smoker vs. charges
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='smoker_yes', y='charges')
plt.title('Smoker vs. Charges')
plt.show()

# Pair plot for multiple variables
sns.pairplot(df, vars=['age', 'bmi', 'children', 'charges'], hue='smoker_yes')
plt.title('Pair plot')
plt.show()

# Statistical tests and relationships

# Calculate average charges for smokers and non-smokers
avg_charges_smoker = df[df['smoker_yes'] == 1]['charges'].mean()
avg_charges_non_smoker = df[df['smoker_yes'] == 0]['charges'].mean()
print(f'Average charges for smokers: {avg_charges_smoker:.2f}')
print(f'Average charges for non-smokers: {avg_charges_non_smoker:.2f}')

# Perform t-test to compare charges for smokers and non-smokers
smoker_charges = df[df['smoker_yes'] == 1]['charges']
non_smoker_charges = df[df['smoker_yes'] == 0]['charges']
t_statistic, p_value = ttest_ind(smoker_charges, non_smoker_charges)
print(f'T-test: t-statistic = {t_statistic:.2f}, p-value = {p_value:.4f}')

# Calculate average charges by number of children
avg_charges_by_children = df.groupby('children')['charges'].mean()
print('Average charges by number of children:')
print(avg_charges_by_children)

# Calculate correlation coefficients
correlation_matrix = df.corr()
correlation_with_charges = correlation_matrix['charges'].sort_values(ascending=False)
print('Correlation with charges:')
print(correlation_with_charges)

# Identify outliers using z-scores
df['charges_zscore'] = zscore(df['charges'])
outliers = df[df['charges_zscore'].abs() > 3]
print('Outliers:')
print(outliers)

# Identify the age range with the highest charges
age_ranges = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100])
charges_by_age_range = df.groupby(age_ranges)['charges'].mean()
highest_charges_age_range = charges_by_age_range.idxmax()
print('Age range with the highest charges:', highest_charges_age_range)
