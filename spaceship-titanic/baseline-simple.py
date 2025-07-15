# %% [markdown]
# # Spaceship Titanic - Simple Baseline Model
#
# Quick baseline with minimal EDA and simple preprocessing to establish baseline performance.
# Compare this against the detailed EDA notebook to see the impact of thorough feature engineering.

# %% [markdown]
# # Libraries

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# # Load Data

# %%
# Load datasets
train_original = pd.read_csv("data/train.csv")
test_original = pd.read_csv("data/test.csv")

print(f"Train shape: {train_original.shape}")
print(f"Test shape: {test_original.shape}")
print(f"Missing values in train:\n{train_original.isnull().sum()}")

# %% [markdown]
# # Clean


# %%
def clean_data(df, is_train=True):
    """Drop columns not needed for modeling"""

    df_processed = df.copy()

    # Drop or standardize columns
    if is_train:
        df_processed = df_processed.drop(columns=["PassengerId"])

    for col in ["Cabin", "Group", "Name"]:
        df_processed = df_processed.drop(columns=[col], errors="ignore")

    return df_processed


train = clean_data(train_original)
test = clean_data(test_original)


# %% [markdown]
# # Quick EDA

# %%
# Target distribution
print("Target distribution:")
print(train_original["Transported"].value_counts(normalize=True))

# Basic statistics
print("\nBasic info:")
print(train_original.info())

# %%
# Check correlation with target for numerical features
numerical_cols = train_original.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Compute correlation matrix (including Transported as int)
corr = train_original.copy()
corr["Transported"] = corr["Transported"].astype(int)
corr_matrix = corr[numerical_cols + ["Transported"]].corr()

# Plot only the lower triangle of the correlation matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, cmap="summer", fmt=".2f", mask=mask)
plt.title("Correlation Heatmap (Lower Triangle)")
plt.show()

# Calculate and print correlations sorted from high to low, formatted in a table
correlations = []
for col in numerical_cols:
    if col in train_original.columns:
        corr_val = train_original[col].corr(train_original["Transported"].astype(int))
        correlations.append((col, corr_val))

# Sort by absolute correlation (high to low)
correlations_sorted = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

# Display as a table
corr_df = pd.DataFrame(correlations_sorted, columns=["Feature", "Correlation"])
print("\nCorrelation with Transported (sorted):")
print(corr_df.to_string(index=False, float_format="%.3f"))


# # %% [markdown]
# # # Feature Engineering
# # Age is your most important feature - create more age-based features


# # %%
# def add_age_features(df):
#     """Enhance age features with additional indicators and groups"""

#     df = df.copy()

#     df["Age_group"] = pd.cut(
#         df["Age"],
#         bins=[0, 18, 25, 100],
#         labels=["Child-Teen", "Young", "Adult"],
#     )

#     # Drop age
#     df = df.drop(columns=["Age"], errors="ignore")

#     return df


# train = add_age_features(train)
# test = add_age_features(test)

# # Plot distribution per age group (KDE off) and show percentiles info
# plt.figure(figsize=(8, 5))
# sns.histplot(
#     data=train_original.assign(
#         Age_group=pd.cut(
#             train_original["Age"],
#             bins=[0, 18, 25, 100],
#             labels=["Child-Teen", "Young", "Adult"],
#         )
#     ),
#     x="Age",
#     hue="Age_group",
#     multiple="stack",
#     kde=False,
#     bins=30,
#     palette="Set2",
# )
# plt.title("Age Distribution by Age Group")
# plt.xlabel("Age")
# plt.ylabel("Count")
# plt.legend(title="Age Group")
# plt.show()

# # Percentiles info per age group
# age_groups = pd.cut(
#     train_original["Age"],
#     bins=[0, 18, 25, 100],
#     labels=["Child-Teen", "Young", "Adult"],
# )
# percentiles = train_original.groupby(age_groups)["Age"].describe(
#     percentiles=[0.25, 0.5, 0.75, 0.9]
# )
# print("\nAge Group Percentiles:")
# print(percentiles[["count", "mean", "25%", "50%", "75%", "90%"]].to_string())

# # %% [markdown]
# # # Spending Features
# # Create total spending feature


# # %%
# def add_spending_features(df):
#     """Create total spending feature and no spending indicator"""

#     df_processed = df.copy()
#     df_processed["Total_spending"] = df_processed[numerical_cols[1:]].sum(axis=1)
#     df_processed["No_spending"] = (df_processed["Total_spending"] == 0).astype(int)

#     # Create spending groups
#     df_processed["Spending_group"] = pd.cut(
#         df_processed["Total_spending"],
#         bins=[-1, 0, 100, 500, 1000, 5000, float("inf")],
#         labels=["No Spending", "Low", "Medium", "High", "Very High", "Luxury"],
#     )

#     return df_processed


# train = add_spending_features(train)
# test = add_spending_features(test)

# # Visualize spending features

# # 1. Distribution of total spending
# plt.figure(figsize=(8, 5))
# sns.histplot(train_spending["Total_spending"], bins=40, color="skyblue")
# plt.title("Distribution of Total Spending")
# plt.xlabel("Total Spending")
# plt.ylabel("Count")
# plt.show()

# # %%
# # 2. Boxplot of total spending by Transported
# plt.figure(figsize=(8, 5))
# sns.boxplot(
#     x=train_original["Transported"], y=train_spending["Total_spending"], palette="Set2"
# )
# plt.title("Total Spending by Transported")
# plt.xlabel("Transported")
# plt.ylabel("Total Spending")
# plt.show()

# # %% [markdown]
# # NOTE:
# # Passengers who were not transported generally spent more on average, with a higher median and more extreme outliers. In contrast, those who were transported tended to spend less, with a lower median and fewer high spenders. This suggests that higher spending is associated with a lower chance of being transported.

# # %%
# # 3. Countplot of spending groups
# plt.figure(figsize=(8, 5))
# sns.countplot(
#     x=train_spending["Spending_group"],
#     order=["No Spending", "Low", "Medium", "High", "Very High", "Luxury"],
# )
# plt.title("Count of Spending Groups")
# plt.xlabel("Spending Group")
# plt.ylabel("Count")
# plt.show()

# # %%
# # 4. Transported rate by spending group
# spending_group_rate = (
#     pd.DataFrame(
#         {
#             "Spending_group": train_spending["Spending_group"],
#             "Transported": train_original["Transported"].astype(int),
#         }
#     )
#     .groupby("Spending_group")["Transported"]
#     .mean()
#     .reindex(["No Spending", "Low", "Medium", "High", "Very High", "Luxury"])
# )

# plt.figure(figsize=(8, 5))
# spending_group_rate.plot(kind="bar", color="coral")
# plt.title("Transported Rate by Spending Group")
# plt.xlabel("Spending Group")
# plt.ylabel("Transported Rate")
# plt.ylim(0, 1)
# plt.show()


# %% [markdown]
# # Simple Preprocessing
# Basic preprocessing with median/mode imputation and simple feature extraction.


# %% [markdown]
# Imputation


# %%
def impute_missing(df):
    df_processed = df.copy()

    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            else:
                df_processed[col] = df_processed[col].fillna(
                    df_processed[col].mode()[0]
                )

    return df_processed


train = impute_missing(train)
test = impute_missing(test)


# %% [markdown]
# # Encoding
# Here we will encode categorical variables using label encoding.

# %%
# Label encode categorical variables
label_encoders = {}
categorical_cols = train.select_dtypes(include=["object", "category"]).columns

for col in categorical_cols:
    if col in train.columns:
        le = LabelEncoder()
        # Fit on combined data to ensure consistent encoding
        combined_data = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined_data.astype(str))

        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

        label_encoders[col] = le

# %%
# Check for NaNs after encoding
print("\nChecking for NaNs after encoding:")

print(train.isnull().sum())
print(test.isnull().sum())

# %% [markdown]
# # Model Training

# %%
X_train = train.copy().drop(columns=["Transported"])
X_test = test.copy()

# Prepare target variable
y_train = train_original["Transported"].astype(int)

print(f"Processed train shape: {X_train.shape}")
print(f"Processed test shape: {X_test.shape}")


# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {X_train_split.shape}")
print(f"Validation set: {X_val_split.shape}")

X_train.head()

# %%
# Train simple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n--- {name} ---")

    # Train model
    model.fit(X_train_split, y_train_split)

    # Validation predictions
    val_pred = model.predict(X_val_split)
    val_score = accuracy_score(y_val_split, val_pred)

    # Cross-validation on full training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    print(f"Validation Accuracy: {val_score:.4f}")
    print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    if val_score > best_score:
        best_score = val_score
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")

# %% [markdown]
# # Feature Importance

# %%
for name, model in models.items():
    if model is not None and hasattr(model, "feature_importances_"):
        print(f"\nFeature importances for {name}:")

        feature_importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\nTop 10 Feature Importances:")
        print(feature_importance.head(10))

    else:
        print("\nNo feature importances available.")

# %% [markdown]
# # Final Model & Predictions

# %%
# Train final model on full training data

if best_model == "Random Forest":
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
elif best_model == "Logistic Regression":
    final_model = LogisticRegression(max_iter=1000, random_state=42)
else:
    raise ValueError("Unsupported model type")

final_model.fit(X_train, y_train)

# Make predictions on test set
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Transported": test_predictions.astype(bool)}
)

submission.to_csv("baseline_submission.csv", index=False)
print(f"\nSubmission file created with {len(submission)} predictions")
print(f"Predicted Transported rate: {test_predictions.mean():.4f}")

# %% [markdown]
# # Summary
#
# **Simple Baseline Results:**
# - Minimal preprocessing (median/mode imputation)
# - Basic feature engineering (group size, deck, total spending)
# - Simple label encoding
# - Random Forest classifier
#
# **Key Insights:**
# - This provides a quick baseline to compare against detailed EDA
# - Shows which features are most important with minimal processing
# - Establishes baseline performance for comparison
#
# **Next Steps:**
# - Compare this baseline against the detailed EDA notebook
# - Identify if sophisticated feature engineering provides significant improvement
# - Focus efforts on areas with biggest impact

# %%
print("=== BASELINE SUMMARY ===")
print(f"Best Model: {best_model_name}")
print(f"Validation Accuracy: {best_score:.4f}")
print(f"Number of features: {X_train.shape[1]}")
print("Training time: Fast (minimal preprocessing)")
print("=========================")
