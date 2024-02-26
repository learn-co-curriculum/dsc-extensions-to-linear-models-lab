# Extensions to Linear Models - Lab

## Introduction

In this lab, you'll practice many concepts you have learned so far, from adding interactions and polynomials to your model to regularization!

## Summary

You will be able to:

- Build a linear regression model with interactions and polynomial features 
- Use feature selection to obtain the optimal subset of features in a dataset

## Let's Get Started!

Below we import all the necessary packages for this lab.


```python
# Run this cell without changes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
```

Load the data.


```python
# Run this cell without changes

# Load data from CSV
df = pd.read_csv("ames.csv")
# Subset columns
df = df[['LotArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF',
         '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd',
         'GarageArea', 'Fireplaces', 'SalePrice']]

# Split the data into X and y
y = df['SalePrice']
X = df.drop(columns='SalePrice')

# Split into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)
```

## Build a Baseline Housing Data Model

Above, we imported the Ames housing data and grabbed a subset of the data to use in this analysis.

Next steps:

- Scale all the predictors using `StandardScaler`, then convert these scaled features back into DataFrame objects
- Build a baseline `LinearRegression` model using *scaled variables* as predictors and use the $R^2$ score to evaluate the model 


```python
# Your code here

# Scale X_train and X_val using StandardScaler

# Ensure X_train and X_val are scaled DataFrames
# (hint: you can set the columns using X.columns)

```


```python
# Your code here

# Create a LinearRegression model and fit it on scaled training data

# Calculate a baseline r-squared score on training data

```

## Add Interactions

Instead of adding all possible interaction terms, let's try a custom technique. We are only going to add the interaction terms that increase the $R^2$ score as much as possible. Specifically we are going to look for the 7 interaction terms that each cause the most increase in the coefficient of determination.

### Find the Best Interactions

Look at all the possible combinations of variables for interactions by adding interactions one by one to the baseline model. Create a data structure that stores the pair of columns used as well as the $R^2$ score for each combination.

***Hint:*** We have imported the `combinations` function from `itertools` for you ([documentation here](https://docs.python.org/3/library/itertools.html#itertools.combinations)). Try applying this to the columns of `X_train` to find all of the possible pairs.

Print the 7 interactions that result in the highest $R^2$ scores.


```python
# Your code here

# Set up data structure


# Find combinations of columns and loop over them

    # Make copies of X_train and X_val
    
    
    # Add interaction term to data

    
    # Find r-squared score (fit on training data, evaluate on validation data)

    
    # Append to data structure
    
    
# Sort and subset the data structure to find the top 7

```

### Add the Best Interactions

Write code to include the 7 most important interactions in `X_train` and `X_val` by adding 7 columns. Use the naming convention `"col1_col2"`, where `col1` and `col2` are the two columns in the interaction.


```python
# Your code here

# Loop over top 7 interactions

    # Extract column names from data structure

    # Construct new column name
    
    # Add new column to X_train and X_val

```

## Add Polynomials

Now let's repeat that process for adding polynomial terms.

### Find the Best Polynomials

Try polynomials of degrees 2, 3, and 4 for each variable, in a similar way you did for interactions (by looking at your baseline model and seeing how $R^2$ increases). Do understand that when going for a polynomial of degree 4 with `PolynomialFeatures`, the particular column is raised to the power of 2 and 3 as well in other terms.

We only want to include "pure" polynomials, so make sure no interactions are included.

Once again you should make a data structure that contains the values you have tested. We recommend a list of tuples of the form:

`(col_name, degree, R2)`, so eg. `('OverallQual', 2, 0.781)` 


```python
# Your code here

# Set up data structure

# Loop over all columns

    # Loop over degrees 2, 3, 4
        
        # Make a copy of X_train and X_val
    
        # Instantiate PolynomialFeatures with relevant degree
        
        # Fit polynomial to column and transform column
        # Hint: use the notation df[[column_name]] to get the right shape
        # Hint: convert the result to a DataFrame
        
        # Add polynomial to data
        # Hint: use pd.concat since you're combining two DataFrames
        # Hint: drop the column before combining so it doesn't appear twice
    
        # Find r-squared score on validation
    
        # Append to data structure

# Sort and subset the data structure to find the top 7

```

### Add the Best Polynomials

If there are duplicate column values in the results above, don't add multiple of them to the model, to avoid creating duplicate columns. (For example, if column `A` appeared in your list as both a 2nd and 3rd degree polynomial, adding both would result in `A` squared being added to the features twice.) Just add in the polynomial that results in the highest R-Squared.


```python
# Your code here

# Filter out duplicates

# Loop over remaining results

    # Create polynomial terms
    
    # Concat new polynomials to X_train and X_val
    
```

Check out your final data set and make sure that your interaction terms as well as your polynomial terms are included.


```python
# Your code here
```

## Full Model R-Squared

Check out the $R^2$ of the full model with your interaction and polynomial terms added. Print this value for both the train and test data.


```python
# Your code here
```

It looks like we may be overfitting some now. Let's try some feature selection techniques.

## Feature Selection

First, test out `RFE` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)) with several different `n_features_to_select` values. For each value, print out the train and validation $R^2$ score and how many features remain.


```python
# Your code here

```

Now test out `Lasso` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)) with several different `alpha` values.


```python
# Your code here

```

Compare the results. Which features would you choose to use?


```python
# Your written answer here
```

## Evaluate Final Model on Test Data

### Data Preparation

At the start of this lab, we created `X_test` and `y_test`. Prepare `X_test` the same way that `X_train` and `X_val` have been prepared. This includes scaling, adding interactions, and adding polynomial terms.


```python
# Your code here

```

### Evaluation

Using either `RFE` or `Lasso`, fit a model on the complete train + validation set, then find R-Squared and MSE values for the test set.


```python
# Your code here

```

## Level Up Ideas (Optional)

### Create a Lasso Path

From this section, you know that when using `Lasso`, more parameters shrink to zero as your regularization parameter goes up. In scikit-learn there is a function `lasso_path()` which visualizes the shrinkage of the coefficients while $alpha$ changes. Try this out yourself!

https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-coordinate-descent-path-py

### AIC and BIC for Subset Selection

This notebook shows how you can use AIC and BIC purely for feature selection. Try this code out on our Ames housing data!

https://xavierbourretsicotte.github.io/subset_selection.html

## Summary

Congratulations! You now know how to apply concepts of bias-variance tradeoff using extensions to linear models and feature selection.
