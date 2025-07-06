# %% [markdown]
# # Titanic Data Science Solutions
# 
# 
# ### This notebook is a companion to the book [Data Science Solutions](https://www.amazon.com/Data-Science-Solutions-Startup-Workflow/dp/1520545312). 
# 
# The notebook walks us through a typical workflow for solving data science competitions at sites like Kaggle.
# 
# There are several excellent notebooks to study data science competition entries. However many will skip some of the explanation on how the solution is developed as these notebooks are developed by experts for experts. The objective of this notebook is to follow a step-by-step workflow, explaining each step and rationale for every decision we take during solution development.
# 
# ## Workflow stages
# 
# The competition solution workflow goes through seven stages described in the Data Science Solutions book.
# 
# 1. Question or problem definition.
# 2. Acquire training and testing data.
# 3. Wrangle, prepare, cleanse the data.
# 4. Analyze, identify patterns, and explore the data.
# 5. Model, predict and solve the problem.
# 6. Visualize, report, and present the problem solving steps and final solution.
# 7. Supply or submit the results.
# 
# The workflow indicates general sequence of how each stage may follow the other. However there are use cases with exceptions.
# 
# - We may combine mulitple workflow stages. We may analyze by visualizing data.
# - Perform a stage earlier than indicated. We may analyze data before and after wrangling.
# - Perform a stage multiple times in our workflow. Visualize stage may be used multiple times.
# - Drop a stage altogether. We may not need supply stage to productize or service enable our dataset for a competition.
# 
# 
# ## Question and problem definition
# 
# Competition sites like Kaggle define the problem to solve or questions to ask while providing the datasets for training your data science model and testing the model results against a test dataset. The question or problem definition for Titanic Survival competition is [described here at Kaggle](https://www.kaggle.com/c/titanic).
# 
# > Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, can our model determine based on a given test dataset not containing the survival information, if these passengers in the test dataset survived or not.
# 
# We may also want to develop some early understanding about the domain of our problem. This is described on the [Kaggle competition description page here](https://www.kaggle.com/c/titanic). Here are the highlights to note.
# 
# - On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.
# - One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.
# - Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# ## Workflow goals
# 
# The data science solutions workflow solves for seven major goals.
# 
# **Classifying.** We may want to classify or categorize our samples. We may also want to understand the implications or correlation of different classes with our solution goal.
# 
# **Correlating.** One can approach the problem based on available features within the training dataset. Which features within the dataset contribute significantly to our solution goal? Statistically speaking is there a [correlation](https://en.wikiversity.org/wiki/Correlation) among a feature and solution goal? As the feature values change does the solution state change as well, and visa-versa? This can be tested both for numerical and categorical features in the given dataset. We may also want to determine correlation among features other than survival for subsequent goals and workflow stages. Correlating certain features may help in creating, completing, or correcting features.
# 
# **Converting.** For modeling stage, one needs to prepare the data. Depending on the choice of model algorithm one may require all features to be converted to numerical equivalent values. So for instance converting text categorical values to numeric values.
# 
# **Completing.** Data preparation may also require us to estimate any missing values within a feature. Model algorithms may work best when there are no missing values.
# 
# **Correcting.** We may also analyze the given training dataset for errors or possibly innacurate values within features and try to corrent these values or exclude the samples containing the errors. One way to do this is to detect any outliers among our samples or features. We may also completely discard a feature if it is not contribting to the analysis or may significantly skew the results.
# 
# **Creating.** Can we create new features based on an existing feature or a set of features, such that the new feature follows the correlation, conversion, completeness goals.
# 
# **Charting.** How to select the right visualization plots and charts depending on nature of the data and the solution goals.
# 
# ### Best practices
# 
# - Performing feature correlation analysis early in the project.
# - Using multiple plots instead of overlays for readability.

# %% [markdown]
# ## Acquire data
# 
# The Python Pandas packages helps us work with our datasets. We start by acquiring the training and testing datasets into Pandas DataFrames. We also combine these datasets to run certain operations on both datasets together.

# %%
import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
combine = [train_df, test_df]

# %% [markdown]
# ## Analyze by describing data
# 
# Pandas also helps describe the datasets answering following questions early in our project.
# 
# **Which features are available in the dataset?**
# 
# Noting the feature names for directly manipulating or analyzing these. These feature names are described on the [Kaggle data page here](https://www.kaggle.com/c/titanic/data).

# %%
print(train_df.columns.values)

# %% [markdown]
# **Which features are categorical?**
# 
# These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.
# 
# - Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
# 
# **Which features are numerical?**
# 
# Which features are numerical? These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.
# 
# - Continous: Age, Fare. Discrete: SibSp, Parch.

# %%
# preview the data
train_df.head()

# %%
train_df.tail()

# %% [markdown]
# **Which features are mixed data types?**
# 
# Numerical, alphanumeric data within same feature. These are candidates for correcting goal.
# 
# - Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
# 
# **Which features may contain errors or typos?**
# 
# This is harder to review for a large dataset, however reviewing a few samples from a smaller dataset may just tell us outright, which features may require correcting.
# 
# - Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.

# %% [markdown]
# **Which features contain blank, null or empty values?**
# 
# These will require correcting.
# 
# - Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
# - Cabin > Age are incomplete in case of test dataset.
# 
# **What are the data types for various features?**
# 
# Helping us during converting goal.
# 
# - Seven features are integer or floats. Six in case of test dataset.
# - Five features are strings (object).

# %%
train_df.info()

# %%
test_df.info()

# %% [markdown]
# **What is the distribution of numerical feature values across the samples?**
# 
# This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.
# 
# - Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
# - Survived is a categorical feature with 0 or 1 values.
# - Around 38% samples survived representative of the actual survival rate at 32%.
# - Most passengers (> 75%) did not travel with parents or children.
# - Nearly 30% of the passengers had siblings and/or spouse aboard.
# - Fares varied significantly with few passengers (<1%) paying as high as $512.
# - Few elderly passengers (<1%) within age range 65-80.

# %% [markdown]
# **What is the distribution of categorical features?**
# 
# - Names are unique across the dataset (count=unique=891)
# - Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
# - Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
# - Embarked takes three possible values. S port used by most passengers (top=S)
# - Ticket feature has high ratio (22%) of duplicate values (unique=681).

# %%
train_df.describe(include=["O"])

# %% [markdown]
# ## Assumtions based on data analysis
# 
# We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.
# 
# **Correlating.**
# 
# We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.
# 
# **Completing.**
# 
# 1. We may want to complete Age feature as it is definitely correlated to survival.
# 2. We may want to complete the Embarked feature as it may also correlate with survival or another important feature.
# 
# **Correcting.**
# 
# 1. Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
# 2. Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
# 3. PassengerId may be dropped from training dataset as it does not contribute to survival.
# 4. Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
# 
# **Creating.**
# 
# 1. We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
# 2. We may want to engineer the Name feature to extract Title as a new feature.
# 3. We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
# 4. We may also want to create a Fare range feature if it helps our analysis.
# 
# **Classifying.**
# 
# We may also add to our assumptions based on the problem description noted earlier.
# 
# 1. Women (Sex=female) were more likely to have survived.
# 2. Children (Age<?) were more likely to have survived. 
# 3. The upper-class passengers (Pclass=1) were more likely to have survived.

# %%
train_df.drop("PassengerId", axis=1).describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`

# %% [markdown]
# ## Analyze by pivoting features
# 
# To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values. It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.
# 
# - **Pclass** We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). We decide to include this feature in our model.
# - **Sex** We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).
# - **SibSp and Parch** These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features (creating #1).

# %%
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(
    by="Survived", ascending=False
)

# %%
train_df[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(
    by="Survived", ascending=False
)

# %%
train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(
    by="Survived", ascending=False
)

# %%
train_df[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(
    by="Survived", ascending=False
)

# %% [markdown]
# ## Analyze by visualizing data
# 
# Now we can continue confirming some of our assumptions using visualizations for analyzing the data.
# 
# ### Correlating numerical features
# 
# Let us start by understanding correlations between numerical features and our solution goal (Survived).
# 
# A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. This helps us answer questions relating to specific bands (Did infants have better survival rate?)
# 
# Note that x-axis in historgram visualizations represents the count of samples or passengers.
# 
# **Observations.**
# 
# - Infants (Age <=4) had high survival rate.
# - Oldest passengers (Age = 80) survived.
# - Large number of 15-25 year olds did not survive.
# - Most passengers are in 15-35 age range.
# 
# **Decisions.**
# 
# This simple analysis confirms our assumptions as decisions for subsequent workflow stages.
# 
# - We should consider Age (our assumption classifying #2) in our model training.
# - Complete the Age feature for null values (completing #1).
# - We should band age groups (creating #3).

# %%
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.FacetGrid(train_df, col="Survived")
g.map(plt.hist, "Age", bins=20)

# %% [markdown]
# ### Correlating numerical and ordinal features
# 
# We can combine multiple features for identifying correlations using a single plot. This can be done with numerical and categorical features which have numeric values.
# 
# **Observations.**
# 
# - Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
# - Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
# - Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
# - Pclass varies in terms of Age distribution of passengers.
# 
# **Decisions.**
# 
# - Consider Pclass for model training.

# %%
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col="Survived", row="Pclass", height=2, aspect=1.5)
grid.map(plt.hist, "Age", alpha=0.5, bins=20)
grid.add_legend();

# %% [markdown]
# ### Correlating categorical features
# 
# Now we can correlate categorical features with our solution goal.
# 
# **Observations.**
# 
# - Female passengers had much better survival rate than males. Confirms classifying (#1).
# - Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
# - Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).
# - Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).
# 
# **Decisions.**
# 
# - Add Sex feature to model training.
# - Complete and add Embarked feature to model training.

# %%
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row="Embarked", height=3, aspect=1.5)
grid.map(sns.pointplot, "Pclass", "Survived", "Sex", palette="deep")
grid.add_legend()

# %% [markdown]
# ### Correlating categorical and numerical features
# 
# We may also want to correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).
# 
# **Observations**
# 
# - Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges.
# - Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).
# 
# **Decisions**
# 
# - Consider banding Fare feature.

# %%
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row="Embarked", col="Survived", height=2, aspect=1.5)
grid.map(sns.barplot, "Sex", "Fare", alpha=0.5)
grid.add_legend()

# %% [markdown]
# ## Wrangle data
# 
# We have collected several assumptions and decisions regarding our datasets and solution requirements. So far we did not have to change a single feature or value to arrive at these. Let us now execute our decisions and assumptions for correcting, creating, and completing goals.

# %%
# # Sort columns by column name in df
# train_df = train_df.sort_index(axis=1)
# train_df.head()

# %% [markdown]
# ### Creating new feature extracting from existing
# 
# TODO: We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# In the following code we extract Title feature using regular expressions. The RegEx pattern `(\w+\.)` matches the first word which ends with a dot character within Name feature. The `expand=False` flag returns a DataFrame.
# We can replace many titles with a more common name or classify them as `Rare`.
# 
# **Observations.**
# 
# When we plot Title, Age, and Survived, we note the following observations.
# 
# - Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
# - Survival among Title Age bands varies slightly.
# - Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
# 
# **Decision.**
# 
# - We decide to retain the new Title feature for model training.

# %%
# # Extract Title as a new column 'TitleExtracted' from Name
# for dataset in combine:
#     dataset["TitleExtracted"] = dataset.Name.str.extract(
#         " ([A-Za-z]+)\\.", expand=False
#     )

# # Map rare titles and standardize as 'TitleMapped'
# for dataset in combine:
#     dataset["TitleMapped"] = dataset["TitleExtracted"].replace(
#         [
#             "Lady",
#             "Countess",
#             "Capt",
#             "Col",
#             "Don",
#             "Dr",
#             "Major",
#             "Rev",
#             "Sir",
#             "Jonkheer",
#             "Dona",
#         ],
#         "Rare",
#     )
#     dataset["TitleMapped"] = dataset["TitleMapped"].replace("Mlle", "Miss")
#     dataset["TitleMapped"] = dataset["TitleMapped"].replace("Ms", "Miss")
#     dataset["TitleMapped"] = dataset["TitleMapped"].replace("Mme", "Mrs")

# # Show crosstab for TitleMapped and Sex
# pd.crosstab(train_df["TitleMapped"], train_df["Sex"])

# %% [markdown]
# We can convert the categorical titles to ordinal.

# %%
# title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
# for dataset in combine:
#     dataset["Title"] = dataset["Title"].map(title_mapping)
#     dataset["Title"] = dataset["Title"].fillna(0)

# train_df.head()

# %% [markdown]
# ### Converting a categorical feature
# 
# Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.

# %% [markdown]
# ### Encoding Sex as a New Feature
# 
# Instead of overwriting the original `Sex` column, we create a new column `SexEncoded` where `female=1` and `male=0`. This preserves the original data for reference and ensures all feature engineering is traceable.

# %%
for dataset in combine:
    dataset["SexEncoded"] = dataset["Sex"].map({"female": 1, "male": 0}).astype(int)

train_df[["Sex", "SexEncoded"]].head()

# %% [markdown]
# ### Filling and Encoding Embarked as New Features
# 
# To preserve the original `Embarked` column, we create a new column `EmbarkedEncoded` where missing values are filled with the most common port and encoded (S=0, C=1, Q=2) for modeling.

# %%
# Fill missing Embarked values as EmbarkedFilled
freq_port = train_df.Embarked.dropna().mode()[0]

# Encode EmbarkedFilled as EmbarkedEncoded
embarked_map = {"S": 0, "C": 1, "Q": 2}
for dataset in combine:
    dataset["EmbarkedEncoded"] = (
        dataset["Embarked"].fillna(freq_port).map(embarked_map).astype(int)
    )

train_df[["Embarked", "EmbarkedEncoded"]].head()

# %% [markdown]
# ### Filling and Binning Fare as New Features
# 
# To preserve the original `Fare` column, we create a new column `FareFilled` where missing values are filled with the median fare. We then create `FareBandEncoded` as a new column representing binned fare groups for modeling.
# 
# We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature. We do this in a single line of code.
# 
# Note that we are not creating an intermediate new feature or doing any further analysis for correlation to guess missing feature as we are replacing only a single value. The completion goal achieves desired requirement for model algorithm to operate on non-null values.
# 
# We may also want round off the fare to two decimals as it represents currency.

# %%
# Fill missing Fare values as FareFilled
for dataset in combine:
    dataset["FareFilled"] = dataset["Fare"].fillna(dataset["Fare"].median())

# Create FareBandEncoded as a new column using quantile binning
train_df["FareBand"], fare_bins = pd.qcut(
    train_df["FareFilled"], 4, retbins=True, labels=False, duplicates="drop"
)
for dataset in combine:
    dataset["FareBandEncoded"] = pd.cut(
        dataset["FareFilled"], bins=fare_bins, labels=False, include_lowest=True
    )

train_df[["Fare", "FareFilled", "FareBandEncoded"]].head()

# %% [markdown]
# ### Creating Engineered Features as New Columns
# 
# We now create new features for modeling, each as a new column:
# - `FamilySize`: total family members aboard (SibSp + Parch + 1)
# - `IsAlone`: 1 if the passenger is alone, 0 otherwise
# - `AgeClassInteraction`: interaction between filled age and class (AgeFilled * Pclass)
# - `AgeBandEncoded` and `FareBandEncoded`: binned/ordinal versions of Age and Fare for modeling consistency

# %%
# Create FamilySize, IsAlone, and AgeClassInteraction as new columns
for dataset in combine:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1
    # dataset["AgeClassInteraction"] = dataset["AgeFilled"] * dataset["Pclass"]

train_df[
    ["SibSp", "Parch", "FamilySize", "IsAlone"]
].head()  # "AgeClassInteraction"

# %% [markdown]
# ### Filling Missing Age and Creating Age Bands as New Features
# 
# Now we should start estimating and completing features with missing or null values. We will first do this for the Age feature.
# 
# We can consider three methods to complete a numerical continuous feature.
# 
# 1. A simple way is to generate random numbers between mean and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation).
# 
# 2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using [median](https://en.wikipedia.org/wiki/Median) values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
# 
# 3. Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.
# 
# Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will prefer method 2.
# 
# To avoid overwriting the original `Age` column, we create a new column `AgeFilled` where missing values are imputed. We then create `AgeBandEncoded` as a new column representing binned age groups. This ensures all transformations are traceable and the original data is preserved.

# %% [markdown]
# TODO: Creating an **AgeBand** (i.e., binning Age into discrete intervals) instead of using the raw Age value has several benefits in the context of the Titanic dataset and many machine learning problems:
# 
# 1. **Handles Non-linearity:**  
#    The relationship between Age and survival may not be linear. For example, children and elderly may have different survival rates compared to adults, regardless of their exact age. Binning helps capture these group effects.
# 
# 2. **Reduces Noise and Outliers:**  
#    Raw Age values can have outliers or small variations that don't matter for survival. Grouping ages smooths out these effects.
# 
# 3. **Simplifies the Model:**  
#    Many algorithms (especially tree-based or categorical models) can work better or faster with fewer, meaningful categories rather than a wide range of continuous values.
# 
# 4. **Deals with Missing Values:**  
#    When you impute missing ages, the exact value may not be accurate. Binning reduces the impact of imputation errors.
# 
# 5. **Improves Interpretability:**  
#    It's easier to explain results like "children under 16 had higher survival" than to interpret a model coefficient for a continuous Age variable.
# 
# **Summary:**  
# Binning Age into bands (creating AgeBand) helps capture non-linear effects, reduces noise, and makes the model more interpretable and robust.
# 
# Let us create Age bands and determine correlations with Survived.

# %%
# Groupwise Age imputation and AgeBandEncoded creation (with AgeFilled)
import numpy as np

guess_ages = np.zeros((2, 3))

for dataset in combine:
    dataset["AgeFilled"] = dataset["Age"]  # Initialize AgeFilled with original Age

    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[
                (dataset["SexEncoded"] == i) & (dataset["Pclass"] == j + 1)
            ]["AgeFilled"].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[
                (dataset["AgeFilled"].isnull())
                & (dataset["SexEncoded"] == i)
                & (dataset["Pclass"] == j + 1),
                "AgeFilled",
            ] = guess_ages[i, j]

    dataset["AgeFilled"] = dataset["AgeFilled"].astype(int)

train_df[["Age", "AgeFilled"]].head()

# %% [markdown]
# We can now remove the AgeBand feature.

# %%
train_df["AgeBand"] = pd.cut(train_df["AgeFilled"], 5)

train_df[["AgeBand", "Survived"]].groupby(
    ["AgeBand"], as_index=False
).mean().sort_values(by="AgeBand", ascending=True)

# %%
for dataset in combine:
    dataset.loc[dataset["AgeFilled"] <= 16, "AgeBandEncoded"] = 0
    dataset.loc[
        (dataset["AgeFilled"] > 16) & (dataset["AgeFilled"] <= 32), "AgeBandEncoded"
    ] = 1
    dataset.loc[
        (dataset["AgeFilled"] > 32) & (dataset["AgeFilled"] <= 48), "AgeBandEncoded"
    ] = 2
    dataset.loc[
        (dataset["AgeFilled"] > 48) & (dataset["AgeFilled"] <= 64), "AgeBandEncoded"
    ] = 3
    dataset.loc[dataset["AgeFilled"] > 64, "AgeBandEncoded"] = 4

train_df[["Age", "AgeFilled", "AgeBandEncoded"]].head()

# %% [markdown]
# ### Create new feature combining existing features
# 
# We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.
# 
# The author created **FamilySize** to capture the total number of family members each passenger was traveling with, since family presence could affect survival chances. However, after analyzing the data, they noticed that simply knowing if a passenger was **alone** or **not alone** (the `IsAlone` feature) was more predictive and easier for the model to use.
# 
# So, they dropped **FamilySize** (and its components `SibSp` and `Parch`) in favor of the simpler **IsAlone** feature, which indicates whether a passenger was traveling alone (`IsAlone = 1`) or with family (`IsAlone = 0`). This reduces complexity and potential noise in the model, focusing on the most relevant information.

# %%
for dataset in combine:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

train_df[["FamilySize", "Survived"]].groupby(
    ["FamilySize"], as_index=False
).mean().sort_values(by="Survived", ascending=False)

# %% [markdown]
# We can create another feature called IsAlone.

# %%
for dataset in combine:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

train_df[["IsAlone", "Survived"]].groupby(["IsAlone"], as_index=False).mean()

# %% [markdown]
# Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.

# %% [markdown]
# Convert the Fare feature to ordinal values based on the FareBand.

# %%
print("Train columns:", train_df.columns.values)
train_df.head()

# %% [markdown]
# And the test dataset.

# %%
test_df.head()

# %%
from datetime import datetime

# Select only derived/modeling columns for export
model_columns = [
    "AgeBandEncoded",
    "AgeClassInteraction",
    "EmbarkedEncoded",
    "FamilySize",
    "FareBandEncoded",
    "IsAlone",
    "Pclass",
    "SexEncoded",
    "Survived",
    "TitleMapped",
    # 'AgeFilled',
    # "Embarked",
    # "FareFilled",
]

# For test set, drop 'Survived' if not present
export_train = train_df[[col for col in model_columns if col in train_df.columns]]
export_test = test_df[[col for col in model_columns if col in test_df.columns]]

# Add to export_test PassengerId for submission
if "PassengerId" in test_df.columns:
    export_test = export_test.join(test_df["PassengerId"])


now = datetime.now().strftime("%Y%m%d_%H%M%S")

train_filename = f"train_processed_{now}.csv"
test_filename = f"test_processed_{now}.csv"

export_train.to_csv(f"data/{train_filename}", index=False)
export_test.to_csv(f"data/{test_filename}", index=False)

# All original columns are preserved in train_df and test_df until this step.


