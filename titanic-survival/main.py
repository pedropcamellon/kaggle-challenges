# %%
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
## Load the data
train_data = pd.read_csv("train.csv")
train_data.head()

# %%
test_data = pd.read_csv("test.csv")
test_data.head()

# %%
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# %%
# Evaluate the model on the training set
y_train_pred = model.predict(X)
train_accuracy = accuracy_score(y, y_train_pred)

print(f"Training Accuracy: {train_accuracy * 100:.2f} %")

# %%
# Cross-validation. To get a more reliable estimate of your model's performance:
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean() * 100:.2f} %")

# %%
# Confusion Matrix. To understand the types of errors your model is making:
cm = confusion_matrix(y, y_train_pred)

print(f"Confusion matrix: {cm}")

sns.heatmap(cm, annot=True, fmt="d")

plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# %%
# Feature Importance. Analyze which features are most influential in model's decisions.

feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
)

feature_importance = feature_importance.sort_values("importance", ascending=False)

print(feature_importance)

# %%
# RandomizedSearchCV
# Define the parameter distribution
param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(1, 20),
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 11),
    "max_features": ["auto", "sqrt", "log2"],
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,  # number of parameter settings sampled
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1,  # use all available cores
)

# Fit the random search object to the data
random_search.fit(X, y)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

model_tuned = random_search.best_estimator_

# %%
# Cross-validation after tuning
cv_scores_after_tuning = cross_val_score(model_tuned, X, y, cv=5)

print(f"Cross-validation scores: {cv_scores_after_tuning}")

cv_scores_after_tuning_mean_cv = cv_scores_after_tuning.mean()

print(f"Mean CV score: {cv_scores_after_tuning_mean_cv * 100:.2f} %")

print(
    f"Mean CV score improvement: {(cv_scores_after_tuning_mean_cv - cv_scores.mean()) * 100:.2f} %"
)

# %%
# Evaluate the model on the test set
predictions = model.predict(X_test)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("submission.csv", index=False)
print("Your submission was successfully saved!")
