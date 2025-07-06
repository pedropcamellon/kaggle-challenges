# Ames House Prices Project

## Overview

This project predicts the sale price of houses in Ames, Iowa using a machine learning pipeline with a Random Forest Regressor. The dataset (`train.csv`) contains various features describing each house.

## Features Used

The following features are used for prediction:

- `GrLivArea`: Above ground living area (square feet)
- `TotalBsmtSF`: Total basement area (square feet)
- `YearBuilt`: Year the house was built
- `OverallQual`: Overall material and finish quality

## Goal

Predict the sale price of houses based on the selected features.

## Approach

1. Load and preprocess the dataset
2. Perform exploratory data analysis (EDA)
3. Select relevant features
4. Apply log transformation to the target variable (`SalePrice`)
5. Train a Random Forest Regressor model
6. Tune hyperparameters using RandomizedSearchCV
7. Evaluate the model's performance using RMSE and cross-validation
8. Make predictions on the test set and generate a submission file

## File Structure

- `main.py`: Main script for data processing, model training, evaluation, and prediction
- `train.csv`: Training dataset
- `test.csv`: Test dataset
- `submission.csv`: Output file for Kaggle submission
- `data_description.txt`: Description of dataset features
- `archive.zip`: Original dataset archive

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scipy`

## Usage

1. Clone the repository
2. Install dependencies using Poetry:

   ```
   poetry install
   ```

3. Run the main script:

   ```
   poetry run python main.py
   ```

   (Run from the `ames-house-prices` directory.)

## Notes

- The model applies a log transformation to the target variable for better performance.
- The final predictions are exponentiated to return to the original price scale.
- You can further improve the model by adding more features or advanced preprocessing.
