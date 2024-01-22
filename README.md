# Salary Prediction Model with Linear Regression

## Overview

This project implements a simple salary prediction model using Linear Regression. The model is trained on a dataset containing information about the years of experience and corresponding salaries. It can be used to predict the salary of employees based on their years of experience.

## Project Structure

- **`salary.csv`**: Dataset containing information about years of experience and salaries.
- **`salary_prediction.py`**: Python script implementing the linear regression model, training, testing, and predicting.

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/salary-prediction.git
    cd salary-prediction
    ```

2. **Install dependencies:**


3. **Run the script:**

    ```bash
    python salary_prediction.py
    ```

    The script will load the dataset, train the linear regression model, and output metrics for evaluating the model on test data. It will also predict the salary for a new employee with a specified number of years of experience.

## Files

- **`salary.csv`**: Dataset file containing information about years of experience and salaries.
- **`salary_prediction.py`**: Python script implementing the linear regression model, training, testing, and predicting.
- **`README.md`**: Documentation file providing an overview of the project and instructions for usage.

## Results

The script generates a scatter plot visualizing the test data along with the linear regression line. Performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) are displayed. The model's ability to predict the salary for a new employee with 7 years of experience is also demonstrated.

