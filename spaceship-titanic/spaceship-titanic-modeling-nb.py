# %% [markdown]
# # Introduction

# %% [markdown]
# Welcome to this comprehensive guide on **binary classification** with the **Spaceship Titanic** dataset. The objective is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.
#
# *We will cover:*
# * Exploratory Data Analysis
# * Feature Engineering
# * Data Cleaning
# * Encoding, Scaling and Preprocessing
# * Training Machine Learning Models
# * Cross Validation and Ensembling Predictions

# %% [markdown]
# # Libraries

# %%
# Core
from imblearn.over_sampling import SMOTE
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
import plotly.express as px
import seaborn as sns
import time
import warnings

warnings.filterwarnings("ignore")

# Sklearn
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)  # plot_confusion_matrix, plot_roc_curve,
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.utils import resample

# Models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# %% [markdown]
# # Data

# %% [markdown]
# **Load data**

# %%
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
data = pd.read_csv("data/cleaned-data.csv")


# Shape and preview
print("Train set shape:", train.shape)
print("Test set shape:", test.shape)
train.head()

# %% [markdown]
# *Feature descriptions:*
# > * **PassengerId** - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# > * **HomePlanet** - The planet the passenger departed from, typically their planet of permanent residence.
# > * **CryoSleep** - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# > * **Cabin** - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# > * **Destination** - The planet the passenger will be debarking to.
# > * **Age** - The age of the passenger.
# > * **VIP** - Whether the passenger has paid for special VIP service during the voyage.
# > * **RoomService**, **FoodCourt**, **ShoppingMall**, **Spa**, **VRDeck** - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# > * **Name** - The first and last names of the passenger.
# > * **Transported** - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# %% [markdown]
# **Missing values**

# %%
print("TRAIN SET MISSING VALUES:")
print(train.isna().sum())
print("---")
print("TEST SET MISSING VALUES:")
print(test.isna().sum())

# %% [markdown]
# # Preprocessing

# %% [markdown]
# **Split data back into train and test sets**

# %%
# Train and test
X = data[data["PassengerId"].isin(train["PassengerId"].values)].copy()
X_test = data[data["PassengerId"].isin(test["PassengerId"].values)].copy()

# %% [markdown]
# **Drop unwanted features**

# %%
# Drop qualitative/redundant/collinear/high cardinality features
X.drop(
    ["PassengerId", "Group", "Group_size", "Age_group", "Cabin_number"],
    axis=1,
    inplace=True,
)
X_test.drop(
    ["Group", "Group_size", "Age_group", "Cabin_number"],
    axis=1,
    inplace=True,
)

# %% [markdown]
# **Log transform**

# %% [markdown]
# The logarithm transform is used to decrease skew in distributions, especially with large outliers. It can make it easier for algorithms to 'learn' the correct relationships. We will apply it to the expenditure features as these are heavily skewed by outliers.

# %%
# Plot log transform results
for i, col in enumerate(
    ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Expenditure"]
):
    fig = plt.figure(figsize=(12, 20))

    plt.subplot(6, 2, 2 * i + 1)
    sns.histplot(X[col], binwidth=100)
    plt.ylim([0, 200])
    plt.title(f"{col} (original)")

    plt.subplot(6, 2, 2 * i + 2)
    sns.histplot(np.log(1 + X[col]), color="C1")
    plt.ylim([0, 200])
    plt.title(f"{col} (log-transform)")

    fig.tight_layout()
    plt.show()

# %%
# Apply log transform
for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Expenditure"]:
    X[col] = np.log(1 + X[col])
    X_test[col] = np.log(1 + X_test[col])

# %% [markdown]
# **Encoding and scaling**

# %% [markdown]
# We will use column transformers to be more professional. It's also good practice.

# %%
# Indentify numerical and categorical columns
numerical_cols = [
    cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
]
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

# Scale numerical data to have mean=0 and variance=1
numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# One-hot encode categorical data
categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                drop="if_binary", handle_unknown="ignore", sparse_output=False
            ),
        )
    ]
)

# Combine preprocessing
ct = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="passthrough",
)

# Apply preprocessing
X = ct.fit_transform(X)
X_test = ct.transform(X_test)

# Print new shape
print("Training set shape:", X.shape)

# %% [markdown]
# **PCA**

# %% [markdown]
# Let's look at the transformed data in PCA space. This gives a low dimensional representation of the data, which preserves local and global structure.

# %%
pca = PCA(n_components=3)
components = pca.fit_transform(X)

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components,
    x=0,
    y=1,
    z=2,
    color=X[:, -1],  # Color by target variable
    color_continuous_scale=px.colors.sequential.Viridis,
    size=0.1 * np.ones(len(X)),
    opacity=1,
    title=f"Total Explained Variance: {total_var:.2f}%",
    labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
    width=800,
    height=500,
)
fig.show()

# %%
# Explained variance (how important each additional principal component is)
pca = PCA().fit(X)
fig, ax = plt.subplots(figsize=(10, 4))
xi = np.arange(1, 1 + X.shape[1], step=1)
yi = np.cumsum(pca.explained_variance_ratio_)
plt.plot(xi, yi, marker="o", linestyle="--", color="b")

# Aesthetics
plt.ylim(0.0, 1.1)
plt.xlabel("Number of Components")
plt.xticks(np.arange(1, 1 + X.shape[1], step=2))
plt.ylabel("Cumulative variance (%)")
plt.title("Explained variance by each component")
plt.axhline(y=1, color="r", linestyle="-")
plt.text(0.5, 0.85, "100% cut-off threshold", color="red")
ax.grid(axis="x")

# %% [markdown]
# **Create a validation set**

# %% [markdown]
# We will use this to choose which model(s) to use.

# %%
# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, stratify=y, train_size=0.8, test_size=0.2, random_state=0
)

# %% [markdown]
# # Model selection

# %% [markdown]
# To briefly mention the algorithms we will use,
#
# **Logistic Regression:** Unlike linear regression which uses Least Squares, this model uses Maximum Likelihood Estimation to fit a sigmoid-curve on the target variable distribution. The sigmoid/logistic curve is commonly used when the data is questions had binary output.
#
# **K-Nearest Neighbors (KNN):** KNN works by selecting the majority class of the k-nearest neighbours, where the metric used is usually Euclidean distance. It is a simple and effective algorithm but can be sensitive by many factors, e.g. the value of k, the preprocessing done to the data and the metric used.
#
# **Support Vector Machine (SVM):** SVM finds the optimal hyperplane that seperates the data in the feature space. Predictions are made by looking at which side of the hyperplane the test point lies on. Ordinary SVM assumes the data is linearly separable, which is not always the case. A kernel trick can be used when this assumption fails to transform the data into a higher dimensional space where it is linearly seperable. SVM is a popular algorithm because it is computationally effecient and produces very good results.
#
# **Random Forest (RF):** RF is a reliable ensemble of decision trees, which can be used for regression or classification problems. Here, the individual trees are built via bagging (i.e. aggregation of bootstraps which are nothing but multiple train datasets created via sampling with replacement) and split using fewer features. The resulting diverse forest of uncorrelated trees exhibits reduced variance; therefore, is more robust towards change in data and carries its prediction accuracy to new data. It works well with both continuous & categorical data.
#
# **Extreme Gradient Boosting (XGBoost):** XGBoost is similar to RF in that it is made up of an ensemble of decision-trees. The difference arises in how those trees as derived; XGboost uses extreme gradient boosting when optimising its objective function. It often produces the best results but is relatively slow compared to other gradient boosting algorithms.
#
# **Light Gradient Boosting Machine (LGBM):** LGBM works essentially the same as XGBoost but with a lighter boosting technique. It usually produces similar results to XGBoost but is significantly faster.
#
# **Categorical Boosting (CatBoost):** CatBoost is an open source algorithm based on gradient boosted decision trees. It supports numerical, categorical and text features. It works well with heterogeneous data and even relatively small data. Informally, it tries to take the best of both worlds from XGBoost and LGBM.
#
# **Naive Bayes (NB):** Naive Bayes learns how to classify samples by using Bayes' Theorem. It uses prior information to 'update' the probability of an event by incoorporateing this information according to Bayes' law. The algorithm is quite fast but a downside is that it assumes the input features are independent, which is not always the case.

# %% [markdown]
# We will train these models and evaluate them on the validation set to then choose which ones to carry through to the next stage (cross validation).

# %% [markdown]
# **Define classifiers**

# %%
# Classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(random_state=0),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(random_state=0, probability=True),
    "RandomForest": RandomForestClassifier(random_state=0),
    # "XGBoost" : XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss'), # XGBoost takes too long
    "LGBM": LGBMClassifier(random_state=0),
    "CatBoost": CatBoostClassifier(random_state=0, verbose=False),
    "NaiveBayes": GaussianNB(),
}

# Grids for grid search
LR_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.25, 0.5, 0.75, 1, 1.25, 1.5],
    "max_iter": [50, 100, 150],
}

KNN_grid = {"n_neighbors": [3, 5, 7, 9], "p": [1, 2]}

SVC_grid = {
    "C": [0.25, 0.5, 0.75, 1, 1.25, 1.5],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
}

RF_grid = {
    "n_estimators": [50, 100, 150, 200, 250, 300],
    "max_depth": [4, 6, 8, 10, 12],
}

boosted_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [4, 8, 12],
    "learning_rate": [0.05, 0.1, 0.15],
}

NB_grid = {"var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7]}

# Dictionary of all grids
grid = {
    "LogisticRegression": LR_grid,
    "KNN": KNN_grid,
    "SVC": SVC_grid,
    "RandomForest": RF_grid,
    "XGBoost": boosted_grid,
    "LGBM": boosted_grid,
    "CatBoost": boosted_grid,
    "NaiveBayes": NB_grid,
}

# %% [markdown]
# **Train and evaluate models**

# %% [markdown]
# Train models with grid search (but no cross validation so it doesn't take too long) to get a rough idea of which are the best models for this dataset.

# %%
i = 0
clf_best_params = classifiers.copy()
valid_scores = pd.DataFrame(
    {
        "Classifer": classifiers.keys(),
        "Validation accuracy": np.zeros(len(classifiers)),
        "Training time": np.zeros(len(classifiers)),
    }
)
for key, classifier in classifiers.items():
    start = time.time()
    clf = GridSearchCV(estimator=classifier, param_grid=grid[key], n_jobs=-1, cv=None)

    # Train and score
    clf.fit(X_train, y_train)
    valid_scores.iloc[i, 1] = clf.score(X_valid, y_valid)

    # Save trained model
    clf_best_params[key] = clf.best_params_

    # Print iteration and training time
    stop = time.time()
    valid_scores.iloc[i, 2] = np.round((stop - start) / 60, 2)

    print("Model:", key)
    print("Training time (mins):", valid_scores.iloc[i, 2])
    print("")
    i += 1

# %%
# Show results
valid_scores

# %% [markdown]
# Motivated by this, we will take LGBM and CatBoost to the final stage of modelling.

# %%
# Show best parameters from grid search
clf_best_params

# %% [markdown]
# # Modelling

# %% [markdown]
# We can finally train our best model on the whole training set using cross validation and ensembling predictions together to produce the most confident predictions.

# %% [markdown]
# **Define best models**

# %%
# Classifiers
best_classifiers = {
    "LGBM": LGBMClassifier(**clf_best_params["LGBM"], random_state=0),
    "CatBoost": CatBoostClassifier(
        **clf_best_params["CatBoost"], verbose=False, random_state=0
    ),
}

# %% [markdown]
# **Cross validation and ensembling predictions**

# %% [markdown]
# Predictions are ensembled together using soft voting. This averages the predicted probabilies to produce the most confident predictions.

# %%
# Number of folds in cross validation
FOLDS = 10

preds = np.zeros(len(X_test))
for key, classifier in best_classifiers.items():
    start = time.time()

    # 10-fold cross validation
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)

    score = 0
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Get training and validation sets
        X_train, X_valid = X[train_idx], X[val_idx]
        y_train, y_valid = y[train_idx], y[val_idx]

        # Train model
        clf = classifier
        clf.fit(X_train, y_train)

        # Make predictions and measure accuracy
        preds += clf.predict_proba(X_test)[:, 1]
        score += clf.score(X_valid, y_valid)

    # Average accuracy
    score = score / FOLDS

    # Stop timer
    stop = time.time()

    # Print accuracy and time
    print("Model:", key)
    print("Average validation accuracy:", np.round(100 * score, 2))
    print("Training time (mins):", np.round((stop - start) / 60, 2))
    print("")

# Ensemble predictions
preds = preds / (FOLDS * len(best_classifiers))

# %% [markdown]
# # Submission

# %% [markdown]
# Let's look at the distribution of the predicted probabilities.

# %%
plt.figure(figsize=(10, 4))
sns.histplot(preds, binwidth=0.01, kde=True)
plt.title("Predicted probabilities")
plt.xlabel("Probability")

# %% [markdown]
# It is interesting to see that the models are either very confident or very unconfident but not much in between.

# %% [markdown]
# **Post processing**

# %% [markdown]
# Finally, we need to convert each predicted probability into one of the two classes (transported or not). The simplest way is to round each probability to the nearest integer (0 for False or 1 for True). However, assuming the train and test sets have similar distributions, we can tune the classification threshold to obtain a similar proportion of transported/not transported in our predictions as in the train set. Remember that the proportion of transported passengers in the train set was 50.4%.

# %%
# Proportion (in test set) we get from rounding
print(np.round(100 * np.round(preds).sum() / len(preds), 2))

# %% [markdown]
# Our models seem to (potentially) overestimate the number of transported passengers in the test set. Let's try to bring that proportion down a bit.


# %%
# Proportion of predicted positive (transported) classes
def preds_prop(preds_arr, thresh):
    pred_classes = (preds_arr >= thresh).astype(int)
    return pred_classes.sum() / len(pred_classes)


# Plot proportions across a range of thresholds
def plot_preds_prop(preds_arr):
    # Array of thresholds
    T_array = np.arange(0, 1, 0.001)

    # Calculate proportions
    prop = np.zeros(len(T_array))
    for i, T in enumerate(T_array):
        prop[i] = preds_prop(preds_arr, T)

    # Plot proportions
    plt.figure(figsize=(10, 4))
    plt.plot(T_array, prop)
    target_prop = 0.519  # Experiment with this value
    plt.axhline(y=target_prop, color="r", linestyle="--")
    plt.text(-0.02, 0.45, f"Target proportion: {target_prop}", fontsize=14)
    plt.title("Predicted target distribution vs threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Proportion")

    # Find optimal threshold (the one that leads to the proportion being closest to target_prop)
    T_opt = T_array[np.abs(prop - target_prop).argmin()]
    print("Optimal threshold:", T_opt)
    return T_opt


T_opt = plot_preds_prop(preds)

# %%
# Classify test set using optimal threshold
preds_tuned = (preds >= T_opt).astype(int)

# %% [markdown]
# **Submit predictions**

# %%
# Sample submission (to get right format)
sub = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")

# Add predictions
sub["Transported"] = preds_tuned

# Replace 0 to False and 1 to True
sub = sub.replace({0: False, 1: True})

# Prediction distribution
plt.figure(figsize=(6, 6))
sub["Transported"].value_counts().plot.pie(
    explode=[0.1, 0.1], autopct="%1.1f%%", shadow=True, textprops={"fontsize": 16}
).set_title("Prediction distribution")

# %%
# Output to csv
sub.to_csv("submission.csv", index=False)
