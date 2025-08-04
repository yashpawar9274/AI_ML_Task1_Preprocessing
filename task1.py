# Importing necessary libraries for data handling, visualization, and preprocessing
import pandas as pd                # For working with datasets (tables)
import numpy as np                 # For numerical operations
import seaborn as sns              # For visualization (like boxplots)
import matplotlib.pyplot as plt    # Another plotting library
from sklearn.preprocessing import StandardScaler  # To scale numeric features

# Step 1: Load the Titanic dataset
df = pd.read_csv('titanic.csv')  # Make sure this file is in the same folder

# Step 2: Get a quick overview of the dataset
print(df.info())            # Shows columns, data types, and missing values
print(df.isnull().sum())    # Shows how many values are missing in each column

# Step 3: Handle missing data
df['Age'].fillna(df['Age'].median(), inplace=True)       # Fill missing ages with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)     # Fill missing fare values (if any)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing ports with most frequent value
df.drop('Cabin', axis=1, inplace=True)  # Drop 'Cabin' because it has too many missing values

# Step 4: Convert text (categorical) data to numbers
# Convert 'Sex' and 'Embarked' columns into dummy variables (binary format)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Step 5: Scale/Normalize 'Age' and 'Fare' to have mean=0 and std=1
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Step 6: Detect and remove outliers from 'Fare' using IQR method
Q1 = df['Fare'].quantile(0.25)  # First quartile
Q3 = df['Fare'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1                   # Interquartile range
# Remove data points where Fare is too far below or above the typical range
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

# Step 7: Save the cleaned and preprocessed dataset to a new CSV file
df.to_csv('cleaned_titanic.csv', index=False)

print("✅ Preprocessing complete. Cleaned data saved as 'cleaned_titanic.csv'")
