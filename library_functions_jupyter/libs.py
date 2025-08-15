import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def transform_y_boxcox(y):
    """
    Apply Box-Cox transformation to the response variable y.
    Ensures y is positive before transformation.
    
    Parameters:
    y (array-like): Response variable
    
    Returns:
    y_transformed (np.ndarray): Box-Cox transformed y
    lambda_val (float): Lambda parameter used in Box-Cox
    """
    y = np.array(y, dtype=float)
    
    # Ensure positivity
    if np.any(y <= 0):
        shift_val = abs(np.min(y)) + 1
        y = y + shift_val
        print(f"Response variable shifted by {shift_val} to make it positive.")
    
    y_transformed, lambda_val = boxcox(y)
    return y_transformed, lambda_val


def residual_plots_y_vs_predictors(X, y, use_column_ids=None):
    """
    Fit a new Linear Regression model with transformed y and 
    plot residuals vs each predictor.
    
    Parameters:
    X (pd.DataFrame or np.ndarray): Feature variables
    y_transformed (array-like): Transformed response variable
    """
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(1, X.shape[1]+1)])
    else:
        X_df = X.copy()

    model = LinearRegression()
    model.fit(X_df, y)
    residuals = y - model.predict(X_df)
    
    # Plot residuals vs each feature
    for i, col in enumerate(X_df.columns):
        if use_column_ids is not None and i not in use_column_ids:
            continue
        plt.figure(figsize=(6, 4))
        plt.scatter(X_df[col], residuals, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel(col)
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs {col} (Box-Cox Transformed Y)")
        plt.show()
