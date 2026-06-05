import pandas as pd

# Step 1: Create DataFrame
data = {
    'Name': ['Dhruv', 'Rahul', 'Priya', 'Amit', 'Dhruvi'],
    'Age': [21, 22, None, 24, 23],
    'Salary': [25000, None, 35000, 45000, 40000],
    'City': ['Vadodara', 'Ahmedabad', 'Surat', None, 'Vadodara']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Step 2: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Fill Missing Values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
df['City'].fillna(df['City'].mode()[0], inplace=True)

# Step 4: Remove Duplicates
df.drop_duplicates(inplace=True)

print("\nPreprocessed Dataset:")
print(df)