# %%
# Import necessary libraries for data manipulation, visualization, and machine learning
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
# Load the training data
# This dataset contains information about Titanic passengers and whether they survived (1) or not (0)
train_data = pd.read_csv("train.csv")
train_data.head()  # Display the first few rows to understand the data structure

# %%
# Load the test data
# This dataset contains similar information but without the survival outcome (to be predicted)
test_data = pd.read_csv("test.csv")
test_data.head()  # Display the first few rows

# %%
# Prepare the target variable and select features for modeling
# 'Survived' is the target we want to predict
# We'll use a few basic features for this baseline model
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])  # Convert categorical variables to numeric
X_test = pd.get_dummies(test_data[features])
y = train_data["Survived"]

# %%
# Train a Random Forest Classifier
# This is an ensemble model that builds multiple decision trees and averages their predictions
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# %%
# Evaluate the model on the training set
# This gives a quick check of how well the model fits the training data
# Note: High training accuracy does not guarantee good performance on new data
y_train_pred = model.predict(X)
train_accuracy = accuracy_score(y, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f} %")

# %%
# Cross-validation: Estimate model performance more reliably
# Splits the data into 5 parts, trains on 4, tests on 1, and repeats
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean() * 100:.2f} %")

# %%
# Confusion Matrix: Visualize prediction errors
# Shows counts of true positives, true negatives, false positives, and false negatives
cm = confusion_matrix(y, y_train_pred)
print(f"Confusion matrix: {cm}")
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix on Training Data")
plt.show()

# %%
# Feature Importance: See which features influence the model's decisions most
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
)
feature_importance = feature_importance.sort_values("importance", ascending=False)
print(feature_importance)

# %%
# Hyperparameter Tuning with RandomizedSearchCV
# Searches for the best combination of model parameters to improve performance
param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(1, 20),
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 11),
    "max_features": ["auto", "sqrt", "log2"],
}
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter settings sampled
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1,  # Use all available cores
)
random_search.fit(X, y)
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)
model_tuned = random_search.best_estimator_

# %%
# Cross-validation after tuning: Check if the tuned model performs better
cv_scores_after_tuning = cross_val_score(model_tuned, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores_after_tuning}")
cv_scores_after_tuning_mean_cv = cv_scores_after_tuning.mean()
print(f"Mean CV score: {cv_scores_after_tuning_mean_cv * 100:.2f} %")
print(
    f"Mean CV score improvement: {(cv_scores_after_tuning_mean_cv - cv_scores.mean()) * 100:.2f} %"
)

# %%
# Make predictions on the test set
# These predictions can be submitted to Kaggle for evaluation
predictions = model.predict(X_test)
output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("submission.csv", index=False)
print("Your submission was successfully saved!")
