# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = sns.load_dataset("diamonds")

# Display basic information
print(df.head())

print(df.info())

print(df.describe())

# %%
numerical_features = [f for f in df.columns if df.dtypes[f] != "category"]

print(f"Numerical features: {numerical_features}")

for col in numerical_features:
    sns.displot(df[col], kde=True)
    plt.show()

# %%
# Pairplot for numerical features
sns.pairplot(df[numerical_features])
plt.show()

# %%
categorical_features = [f for f in df.columns if df.dtypes[f] == "category"]

print(f"Categorical features: {categorical_features}")

for col in categorical_features:
    sns.displot(df[col])
    plt.show()

# %%
# Boxplots for categorical features
for col in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y="price", data=df)
    plt.title(f"Price distribution by {col}")
    plt.show()

# %%
# Calculate volume
df["volume"] = df["x"] * df["y"] * df["z"]

# Create price per carat feature
df["price_per_carat"] = df["price"] / df["carat"]

sns.pairplot(df[["price", "carat", "volume", "price_per_carat"]])
plt.show()
