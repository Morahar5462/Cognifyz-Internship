import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Ds.csv')

# --- Task 1: Table Booking and Online Delivery ---
print("--- Task 1: Table Booking and Online Delivery ---")

# Convert 'Has Table booking' and 'Has Online delivery' to boolean
df['Has Table booking'] = df['Has Table booking'].map({'Yes': True, 'No': False})
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': True, 'No': False})

# Calculate the percentage of restaurants that offer table booking
table_booking_percentage = df['Has Table booking'].mean() * 100
print(f"Percentage of restaurants with table booking: {table_booking_percentage:.2f}%")

# Calculate the percentage of restaurants that offer online delivery
online_delivery_percentage = df['Has Online delivery'].mean() * 100
print(f"Percentage of restaurants with online delivery: {online_delivery_percentage:.2f}%")

# Compare average ratings of restaurants with and without table booking
avg_rating_with_booking = df[df['Has Table booking'] == True]['Aggregate rating'].mean()
avg_rating_without_booking = df[df['Has Table booking'] == False]['Aggregate rating'].mean()
print(f"Average rating with table booking: {avg_rating_with_booking:.2f}")
print(f"Average rating without table booking: {avg_rating_without_booking:.2f}")

# Analyze availability of online delivery among restaurants with different price ranges
delivery_by_price_range = df.groupby('Price range')['Has Online delivery'].value_counts(normalize=True) * 100
print("\nOnline delivery availability by price range:")
print(delivery_by_price_range)

# --- Task 2: Price Range Analysis ---
print("\n--- Task 2: Price Range Analysis ---")

# Determine the most common price range
most_common_price_range = df['Price range'].mode()[0]
print(f"Most common price range: {most_common_price_range}")

# Calculate the average rating for each price range
avg_rating_by_price_range = df.groupby('Price range')['Aggregate rating'].mean()
print("\nAverage rating by price range:")
print(avg_rating_by_price_range)

# Identify the color that represents the highest average rating among different price ranges
# Create a mapping from price range to rating color
price_range_colors = df.groupby('Price range')['Rating color'].apply(lambda x: x.mode()[0])

# Get the price range with the highest average rating
best_price_range = avg_rating_by_price_range.idxmax()

# Get the color for the best price range
best_color = price_range_colors[best_price_range]

print(f"\nThe color that represents the highest average rating (price range {best_price_range}): {best_color}")

# --- Task 3: Feature Engineering ---
print("\n--- Task 3: Feature Engineering ---")

# Extract length of restaurant name
df['Restaurant Name Length'] = df['Restaurant Name'].str.len()

# Extract length of address
df['Address Length'] = df['Address'].str.len()

# Display the first few rows with the new features
print("\nFirst few rows with new features:")
print(df[['Restaurant Name', 'Restaurant Name Length', 'Address', 'Address Length']].head())

#Convert boolean values to 0 and 1
df['Has Table booking'] = df['Has Table booking'].astype(int)
df['Has Online delivery'] = df['Has Online delivery'].astype(int)
print("\nFirst few rows with encoded features:")
print(df[['Has Table booking', 'Has Online delivery']].head())
