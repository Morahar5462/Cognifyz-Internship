import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Ds.csv')

# --- Task 1: Data Exploration and Preprocessing ---

# 1.  Dataset Dimensions
print("Task 1:\n")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# 2.  Missing Values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values (example: fill missing cuisines with 'Unknown')
df['Cuisines'] = df['Cuisines'].fillna('Unknown')

#Verify Missing Values are Handled
print("\nMissing values after handling:")
print(df.isnull().sum())


# 3.  Data Type Conversion (if necessary)
print("\nData types:")
print(df.dtypes)

#No immediate data type conversions appear necessary based on initial inspection

# 4.  Target Variable Analysis ("Aggregate rating")
print("\nTarget Variable Analysis:")
print(df['Aggregate rating'].value_counts())

# Plot distribution of Aggregate rating
plt.figure(figsize=(10, 6))
sns.histplot(df['Aggregate rating'], bins=30, kde=True)
plt.title('Distribution of Aggregate Rating')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.show()

print("\n--- Task 2: Descriptive Analysis ---")

# 1.  Statistical Measures for Numerical Columns
print("\nStatistical Measures for Numerical Columns:")
numerical_cols = ['Average Cost for two', 'Longitude', 'Latitude', 'Votes','Price range']
print(df[numerical_cols].describe())

# 2.  Distribution of Categorical Variables
print("\nDistribution of Categorical Variables:")
categorical_cols = ['Country Code', 'City', 'Cuisines']

for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts().head(10))  # Show top 10 for brevity

# 3.  Top Cuisines and Cities
print("\nTop Cuisines and Cities:")
print("\nTop 10 Cuisines:")
print(df['Cuisines'].value_counts().head(10))

print("\nTop 10 Cities:")
print(df['City'].value_counts().head(10))

print("\n--- Task 3: Geospatial Analysis ---")

# 1.  Restaurant Locations on a Map (using scatter plot as a simple visualization)
print("\nRestaurant Locations on a Map (Scatter Plot):")
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Longitude', y='Latitude', hue='Aggregate rating', data=df, palette='viridis')
plt.title('Restaurant Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# 2.  Restaurant Distribution (Distribution across cities)
print("\nRestaurant Distribution Across Cities:")
city_counts = df['City'].value_counts()
print(city_counts)

# 3.  Correlation Analysis (Location and Rating)
print("\nCorrelation Analysis (Location and Rating):")
correlation = df[['Longitude', 'Latitude', 'Aggregate rating']].corr()
print(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Location and Rating')
plt.show()
