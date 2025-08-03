# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Extract - Load data from a CSV file
df = pd.read_csv('your_dataset.csv')  # Replace with your actual file name

print("\n--- Raw Data Sample ---")
print(df.head())

# Step 2: Transform

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Normalize numeric columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Add formatted date column (if not present)
if 'date' not in df.columns:
    df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')

# Step 3: Load - Save cleaned/transformed data to new CSV
df.to_csv('processed_data.csv', index=False)

# Output
print("\n--- Processed Data Sample ---")
print(df.head())
