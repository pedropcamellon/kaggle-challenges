# %%
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
    train_test_split,
)

# %%
# Load train data
train_data = pd.read_csv("train.csv")
train_data.head()

# %%
# Exploratory Data Analysis

# %%
# Data Preprocessing
# Handle missing values
# Encode categorical variables
# Scale numerical features

# %%
# Feature selection
features = ["GrLivArea", "TotalBsmtSF", "YearBuilt", "OverallQual"]

X = train_data[features]
y = np.log(train_data["SalePrice"])  # Log transform the target variable

# %%
# Model Training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
# Model Evaluation
y_val_pred = model.predict(X_val)

val_mse = mean_squared_error(y_val, y_val_pred)

print(f"Validation MSE: {val_mse:.4f}")

val_rmse = np.sqrt(val_mse)

print(f"Validation RMSE: {val_rmse:.4f}")

val_rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(y_val_pred)))

print(f"Validation RMSE (in dollars): ${val_rmse:.2f}")

# %%
# Hyperparameter Tuning
param_dist = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(5, 20),
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 11),
}

random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,  # use all available cores
)

random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

model_tuned = random_search.best_estimator_

# %%
# Evaluate best model on validation set
y_val_pred_best = model_tuned.predict(X_val)
val_rmse_best = np.sqrt(mean_squared_error(y_val, y_val_pred_best))

print(f"Best Model Validation RMSE: {val_rmse_best:.4f}")

# %%
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

final_model = RandomForestRegressor(**random_search.best_params_, random_state=42)
final_model.fit(X_full, y_full)

# %%
# Cross-validation
X_full = pd.concat([X_train, X_val])

y_full = pd.concat([y_train, y_val])

cv_scores = cross_val_score(
    final_model, X_full, y_full, cv=5, scoring="neg_mean_squared_error"
)

rmse_scores = np.sqrt(-cv_scores)

print(
    f"Cross-validation RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})"
)

# %%
# Load test data
test_data = pd.read_csv("test.csv")
test_data.head()

# %%
# Make predictions and create submission file
X_test = test_data[features]

y_test_pred = np.exp(
    final_model.predict(X_test)
)  # Remember to exponentiate the predictions

submission = pd.DataFrame({"Id": test_data.Id, "SalePrice": y_test_pred})
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
