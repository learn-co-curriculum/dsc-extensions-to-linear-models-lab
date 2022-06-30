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


```python
# __SOLUTION__ 
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
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0)
```


```python
# __SOLUTION__ 

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
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0)
```

## Build a Baseline Housing Data Model

Above, we imported the Ames housing data and grabbed a subset of the data to use in this analysis.

Next steps:

- Scale all the predictors using `StandardScaler`, then convert these scaled features back into DataFrame objects
- Build a baseline `LinearRegression` model using *scaled variables* as predictors and use the $R^2$ score to evaluate the model 


```python
# Your code here

# Scale X_train and X_test using StandardScaler

# Ensure X_train and X_test are scaled DataFrames
# (hint: you can set the columns using X.columns)

```


```python
# __SOLUTION__

# Scale X_train and X_test using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure X_train and X_test are scaled DataFrames
# (hint: you can set the columns using X.columns)
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>GrLivArea</th>
      <th>TotRmsAbvGrd</th>
      <th>GarageArea</th>
      <th>Fireplaces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.114710</td>
      <td>-0.099842</td>
      <td>-0.509252</td>
      <td>-0.639316</td>
      <td>-0.804789</td>
      <td>1.261552</td>
      <td>0.499114</td>
      <td>0.250689</td>
      <td>0.327629</td>
      <td>-0.994820</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.176719</td>
      <td>0.632038</td>
      <td>-0.509252</td>
      <td>0.838208</td>
      <td>0.641608</td>
      <td>-0.808132</td>
      <td>-0.247249</td>
      <td>-0.365525</td>
      <td>0.079146</td>
      <td>-0.994820</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.246336</td>
      <td>-0.831723</td>
      <td>1.304613</td>
      <td>-0.012560</td>
      <td>-0.329000</td>
      <td>-0.808132</td>
      <td>-0.944766</td>
      <td>-0.981739</td>
      <td>-1.105931</td>
      <td>-0.994820</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.378617</td>
      <td>-0.831723</td>
      <td>1.304613</td>
      <td>-0.339045</td>
      <td>-0.609036</td>
      <td>-0.808132</td>
      <td>-1.146010</td>
      <td>-0.981739</td>
      <td>-1.134602</td>
      <td>0.588023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.010898</td>
      <td>-1.563603</td>
      <td>1.304613</td>
      <td>-2.531499</td>
      <td>-1.315922</td>
      <td>0.550523</td>
      <td>-0.481708</td>
      <td>0.250689</td>
      <td>-2.281450</td>
      <td>-0.994820</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>816</th>
      <td>-0.532331</td>
      <td>-0.099842</td>
      <td>-0.509252</td>
      <td>-0.510628</td>
      <td>-0.897228</td>
      <td>-0.808132</td>
      <td>-1.353116</td>
      <td>-2.214167</td>
      <td>-0.274466</td>
      <td>0.588023</td>
    </tr>
    <tr>
      <th>817</th>
      <td>-0.309245</td>
      <td>-0.099842</td>
      <td>-0.509252</td>
      <td>0.514106</td>
      <td>0.315353</td>
      <td>-0.808132</td>
      <td>-0.481708</td>
      <td>-0.365525</td>
      <td>0.088703</td>
      <td>-0.994820</td>
    </tr>
    <tr>
      <th>818</th>
      <td>0.119419</td>
      <td>0.632038</td>
      <td>-0.509252</td>
      <td>-0.513011</td>
      <td>-0.899947</td>
      <td>1.684999</td>
      <td>0.796096</td>
      <td>0.866903</td>
      <td>-0.207566</td>
      <td>0.588023</td>
    </tr>
    <tr>
      <th>819</th>
      <td>-0.002718</td>
      <td>-0.099842</td>
      <td>1.304613</td>
      <td>-0.889542</td>
      <td>-1.329516</td>
      <td>0.783758</td>
      <td>-0.290233</td>
      <td>-0.365525</td>
      <td>-0.852668</td>
      <td>-0.994820</td>
    </tr>
    <tr>
      <th>820</th>
      <td>0.086287</td>
      <td>-0.099842</td>
      <td>0.397681</td>
      <td>0.433080</td>
      <td>0.179414</td>
      <td>-0.808132</td>
      <td>-0.579400</td>
      <td>-0.365525</td>
      <td>-0.675863</td>
      <td>2.170867</td>
    </tr>
  </tbody>
</table>
<p>821 rows × 10 columns</p>
</div>




```python
# Your code here

# Create a LinearRegression model and fit it on scaled training data

# Calculate a baseline r-squared score on training data

```


```python
# __SOLUTION__ 

# Create a LinearRegression model and fit it on scaled training data
regression = LinearRegression()
regression.fit(X_train, y_train)

# Calculate a baseline r-squared score on training data
baseline = regression.score(X_train, y_train)
baseline
```




    0.7868344817421309



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

    # Make copies of X_train and X_test
    
    
    # Add interaction term to data

    
    # Find r-squared score (fit on training data, evaluate on test data)

    
    # Append to data structure
    
    
# Sort and subset the data structure to find the top 7

```


```python
# __SOLUTION__

# Set up data structure
# (Here we are using a list of tuples, but you could use a dictionary,
# a list of lists, some other structure. Whatever makes sense to you.)
interactions = []

# Find combinations of columns and loop over them
column_pairs = list(combinations(X_train.columns, 2))
for (col1, col2) in column_pairs:
    # Make copies of X_train and X_test
    features_train = X_train.copy()
    features_test = X_test.copy()
    
    # Add interaction term to data
    features_train["interaction"] = features_train[col1] * features_train[col2]
    features_test["interaction"] = features_test[col1] * features_test[col2]
    
    # Find r-squared score (fit on training data, evaluate on test data)
    score = LinearRegression().fit(features_train, y_train).score(features_test, y_test)
    
    # Append to data structure
    interactions.append(((col1, col2), score))

# Sort and subset the data structure to find the top 7
top_7_interactions = sorted(interactions, key=lambda record: record[1], reverse=True)[:7]
print("Top 7 interactions:")
print(top_7_interactions)
```

    Top 7 interactions:
    [(('LotArea', '1stFlrSF'), 0.7211105666140574), (('LotArea', 'TotalBsmtSF'), 0.7071649207050104), (('LotArea', 'GrLivArea'), 0.6690980823779029), (('LotArea', 'Fireplaces'), 0.6529699515652587), (('2ndFlrSF', 'TotRmsAbvGrd'), 0.647299489040519), (('OverallCond', 'TotalBsmtSF'), 0.6429019879233769), (('OverallQual', '2ndFlrSF'), 0.6422324294284367)]


### Add the Best Interactions

Write code to include the 7 most important interactions in `X_train` and `X_test` by adding 7 columns. Use the naming convention `"col1_col2"`, where `col1` and `col2` are the two columns in the interaction.


```python
# Your code here

# Loop over top 7 interactions

    # Extract column names from data structure

    # Construct new column name
    
    # Add new column to X_train and X_test

```


```python
# __SOLUTION__ 

# Loop over top 7 interactions
for record in top_7_interactions:
    # Extract column names from data structure
    col1, col2 = record[0]
    
    # Construct new column name
    new_col_name = col1 + "_" + col2
    
    # Add new column to X_train and X_test
    X_train[new_col_name] = X_train[col1] * X_train[col2]
    X_test[new_col_name] = X_test[col1] * X_test[col2]
    
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>GrLivArea</th>
      <th>TotRmsAbvGrd</th>
      <th>GarageArea</th>
      <th>Fireplaces</th>
      <th>LotArea_1stFlrSF</th>
      <th>LotArea_TotalBsmtSF</th>
      <th>LotArea_GrLivArea</th>
      <th>LotArea_Fireplaces</th>
      <th>2ndFlrSF_TotRmsAbvGrd</th>
      <th>OverallCond_TotalBsmtSF</th>
      <th>OverallQual_2ndFlrSF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.114710</td>
      <td>-0.099842</td>
      <td>-0.509252</td>
      <td>-0.639316</td>
      <td>-0.804789</td>
      <td>1.261552</td>
      <td>0.499114</td>
      <td>0.250689</td>
      <td>0.327629</td>
      <td>-0.994820</td>
      <td>0.092318</td>
      <td>0.073336</td>
      <td>-0.057254</td>
      <td>0.114116</td>
      <td>0.316257</td>
      <td>0.325573</td>
      <td>-0.125956</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.176719</td>
      <td>0.632038</td>
      <td>-0.509252</td>
      <td>0.838208</td>
      <td>0.641608</td>
      <td>-0.808132</td>
      <td>-0.247249</td>
      <td>-0.365525</td>
      <td>0.079146</td>
      <td>-0.994820</td>
      <td>-0.113385</td>
      <td>-0.148128</td>
      <td>0.043694</td>
      <td>0.175804</td>
      <td>0.295392</td>
      <td>-0.426859</td>
      <td>-0.510770</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.246336</td>
      <td>-0.831723</td>
      <td>1.304613</td>
      <td>-0.012560</td>
      <td>-0.329000</td>
      <td>-0.808132</td>
      <td>-0.944766</td>
      <td>-0.981739</td>
      <td>-1.105931</td>
      <td>-0.994820</td>
      <td>0.081045</td>
      <td>0.003094</td>
      <td>0.232730</td>
      <td>0.245060</td>
      <td>0.793375</td>
      <td>-0.016386</td>
      <td>0.672141</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.378617</td>
      <td>-0.831723</td>
      <td>1.304613</td>
      <td>-0.339045</td>
      <td>-0.609036</td>
      <td>-0.808132</td>
      <td>-1.146010</td>
      <td>-0.981739</td>
      <td>-1.134602</td>
      <td>0.588023</td>
      <td>0.230591</td>
      <td>0.128368</td>
      <td>0.433899</td>
      <td>-0.222636</td>
      <td>0.793375</td>
      <td>-0.442323</td>
      <td>0.672141</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.010898</td>
      <td>-1.563603</td>
      <td>1.304613</td>
      <td>-2.531499</td>
      <td>-1.315922</td>
      <td>0.550523</td>
      <td>-0.481708</td>
      <td>0.250689</td>
      <td>-2.281450</td>
      <td>-0.994820</td>
      <td>0.014341</td>
      <td>0.027589</td>
      <td>0.005250</td>
      <td>0.010842</td>
      <td>0.138010</td>
      <td>-3.302627</td>
      <td>-0.860799</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>816</th>
      <td>-0.532331</td>
      <td>-0.099842</td>
      <td>-0.509252</td>
      <td>-0.510628</td>
      <td>-0.897228</td>
      <td>-0.808132</td>
      <td>-1.353116</td>
      <td>-2.214167</td>
      <td>-0.274466</td>
      <td>0.588023</td>
      <td>0.477622</td>
      <td>0.271823</td>
      <td>0.720306</td>
      <td>-0.313023</td>
      <td>1.789339</td>
      <td>0.260039</td>
      <td>0.080686</td>
    </tr>
    <tr>
      <th>817</th>
      <td>-0.309245</td>
      <td>-0.099842</td>
      <td>-0.509252</td>
      <td>0.514106</td>
      <td>0.315353</td>
      <td>-0.808132</td>
      <td>-0.481708</td>
      <td>-0.365525</td>
      <td>0.088703</td>
      <td>-0.994820</td>
      <td>-0.097522</td>
      <td>-0.158985</td>
      <td>0.148966</td>
      <td>0.307643</td>
      <td>0.295392</td>
      <td>-0.261809</td>
      <td>0.080686</td>
    </tr>
    <tr>
      <th>818</th>
      <td>0.119419</td>
      <td>0.632038</td>
      <td>-0.509252</td>
      <td>-0.513011</td>
      <td>-0.899947</td>
      <td>1.684999</td>
      <td>0.796096</td>
      <td>0.866903</td>
      <td>-0.207566</td>
      <td>0.588023</td>
      <td>-0.107471</td>
      <td>-0.061263</td>
      <td>0.095069</td>
      <td>0.070221</td>
      <td>1.460730</td>
      <td>0.261252</td>
      <td>1.064983</td>
    </tr>
    <tr>
      <th>819</th>
      <td>-0.002718</td>
      <td>-0.099842</td>
      <td>1.304613</td>
      <td>-0.889542</td>
      <td>-1.329516</td>
      <td>0.783758</td>
      <td>-0.290233</td>
      <td>-0.365525</td>
      <td>-0.852668</td>
      <td>-0.994820</td>
      <td>0.003613</td>
      <td>0.002418</td>
      <td>0.000789</td>
      <td>0.002704</td>
      <td>-0.286483</td>
      <td>-1.160508</td>
      <td>-0.078252</td>
    </tr>
    <tr>
      <th>820</th>
      <td>0.086287</td>
      <td>-0.099842</td>
      <td>0.397681</td>
      <td>0.433080</td>
      <td>0.179414</td>
      <td>-0.808132</td>
      <td>-0.579400</td>
      <td>-0.365525</td>
      <td>-0.675863</td>
      <td>2.170867</td>
      <td>0.015481</td>
      <td>0.037369</td>
      <td>-0.049995</td>
      <td>0.187318</td>
      <td>0.295392</td>
      <td>0.172228</td>
      <td>0.080686</td>
    </tr>
  </tbody>
</table>
<p>821 rows × 17 columns</p>
</div>



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
        
        # Make a copy of X_train and X_test
    
        # Instantiate PolynomialFeatures with relevant degree
        
        # Fit polynomial to column and transform column
        # Hint: use the notation df[[column_name]] to get the right shape
        # Hint: convert the result to a DataFrame
        
        # Add polynomial to data
        # Hint: use pd.concat since you're combining two DataFrames
        # Hint: drop the column before combining so it doesn't appear twice
    
        # Find r-squared score
    
        # Append to data structure

# Sort and subset the data structure to find the top 7

```


```python
# __SOLUTION__ 

# Set up data structure
polynomials = []

# Loop over all columns
for col in X_train.columns:
    # Loop over degrees 2, 3, 4
    for degree in (2, 3, 4):
        
        # Make a copy of X_train and X_test
        features_train = X_train.copy().reset_index()
        features_test = X_test.copy().reset_index()
    
        # Instantiate PolynomialFeatures with relevant degree
        poly = PolynomialFeatures(degree, include_bias=False)
        
        # Fit polynomial to column and transform column
        # Hint: use the notation df[[column_name]] to get the right shape
        # Hint: convert the result to a DataFrame
        col_transformed_train = pd.DataFrame(poly.fit_transform(features_train[[col]]))
        col_transformed_test = pd.DataFrame(poly.transform(features_test[[col]]))
        
        # Add polynomial to data
        # Hint: use pd.concat since you're combining two DataFrames
        # Hint: drop the column before combining so it doesn't appear twice
        features_train = pd.concat([features_train.drop(col, axis=1), col_transformed_train], axis=1)
        features_test = pd.concat([features_test.drop(col, axis=1), col_transformed_test], axis=1)
    
        # Find r-squared score
        score = LinearRegression().fit(features_train, y_train).score(features_test, y_test)
    
        # Append to data structure
        polynomials.append((col, degree, score))
    
# Sort and subset the data structure to find the top 7
top_7_polynomials = sorted(polynomials, key=lambda record: record[-1], reverse=True)[:7]
print("Top 7 polynomials:")
print(top_7_polynomials)
```

    Top 7 polynomials:
    [('GrLivArea', 3, 0.8283186230750728), ('OverallQual_2ndFlrSF', 3, 0.8221477940922196), ('LotArea_Fireplaces', 4, 0.8124290394772224), ('LotArea_Fireplaces', 3, 0.8122028721735164), ('OverallQual', 3, 0.8068150958879932), ('OverallQual_2ndFlrSF', 2, 0.8057158750082858), ('OverallQual', 4, 0.8033460977380442)]


### Add the Best Polynomials

If there are duplicate column values in the results above, don't add multiple of them to the model, to avoid creating duplicate columns. (For example, if column `A` appeared in your list as both a 2nd and 3rd degree polynomial, adding both would result in `A` squared being added to the features twice.) Just add in the polynomial that results in the highest R-Squared.


```python
# Your code here

# Filter out duplicates

# Loop over remaining results

    # Create polynomial terms
    
    # Concat new polynomials to X_train and X_test
    
```


```python
# __SOLUTION__
# Convert to DataFrame for easier manipulation
top_polynomials = pd.DataFrame(top_7_polynomials, columns=["Column", "Degree", "R^2"])
top_polynomials
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Degree</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GrLivArea</td>
      <td>3</td>
      <td>0.828319</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OverallQual_2ndFlrSF</td>
      <td>3</td>
      <td>0.822148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LotArea_Fireplaces</td>
      <td>4</td>
      <td>0.812429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LotArea_Fireplaces</td>
      <td>3</td>
      <td>0.812203</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OverallQual</td>
      <td>3</td>
      <td>0.806815</td>
    </tr>
    <tr>
      <th>5</th>
      <td>OverallQual_2ndFlrSF</td>
      <td>2</td>
      <td>0.805716</td>
    </tr>
    <tr>
      <th>6</th>
      <td>OverallQual</td>
      <td>4</td>
      <td>0.803346</td>
    </tr>
  </tbody>
</table>
</div>




```python
# __SOLUTION__
# Drop duplicate columns
top_polynomials.drop_duplicates(subset="Column", inplace=True)
top_polynomials
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Degree</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GrLivArea</td>
      <td>3</td>
      <td>0.828319</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OverallQual_2ndFlrSF</td>
      <td>3</td>
      <td>0.822148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LotArea_Fireplaces</td>
      <td>4</td>
      <td>0.812429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OverallQual</td>
      <td>3</td>
      <td>0.806815</td>
    </tr>
  </tbody>
</table>
</div>




```python
# __SOLUTION__

# Loop over remaining results
for (col, degree, _) in top_polynomials.values:
    # Create polynomial terms
    poly = PolynomialFeatures(degree, include_bias=False)
    col_transformed_train = pd.DataFrame(poly.fit_transform(X_train[[col]]),
                                        columns=poly.get_feature_names([col]))
    col_transformed_test = pd.DataFrame(poly.transform(X_test[[col]]),
                                    columns=poly.get_feature_names([col]))
    # Concat new polynomials to X_train and X_test
    X_train = pd.concat([X_train.drop(col, axis=1), col_transformed_train], axis=1)
    X_test = pd.concat([X_test.drop(col, axis=1), col_transformed_test], axis=1)
    
```

Check out your final data set and make sure that your interaction terms as well as your polynomial terms are included.


```python
# Your code here
```


```python
# __SOLUTION__ 
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotArea</th>
      <th>OverallCond</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>TotRmsAbvGrd</th>
      <th>GarageArea</th>
      <th>Fireplaces</th>
      <th>LotArea_1stFlrSF</th>
      <th>LotArea_TotalBsmtSF</th>
      <th>...</th>
      <th>OverallQual_2ndFlrSF</th>
      <th>OverallQual_2ndFlrSF^2</th>
      <th>OverallQual_2ndFlrSF^3</th>
      <th>LotArea_Fireplaces</th>
      <th>LotArea_Fireplaces^2</th>
      <th>LotArea_Fireplaces^3</th>
      <th>LotArea_Fireplaces^4</th>
      <th>OverallQual</th>
      <th>OverallQual^2</th>
      <th>OverallQual^3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.114710</td>
      <td>-0.509252</td>
      <td>-0.639316</td>
      <td>-0.804789</td>
      <td>1.261552</td>
      <td>0.250689</td>
      <td>0.327629</td>
      <td>-0.994820</td>
      <td>0.092318</td>
      <td>0.073336</td>
      <td>...</td>
      <td>-0.125956</td>
      <td>0.015865</td>
      <td>-0.001998</td>
      <td>0.114116</td>
      <td>0.013022</td>
      <td>0.001486</td>
      <td>1.695855e-04</td>
      <td>-0.099842</td>
      <td>0.009968</td>
      <td>-0.000995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.176719</td>
      <td>-0.509252</td>
      <td>0.838208</td>
      <td>0.641608</td>
      <td>-0.808132</td>
      <td>-0.365525</td>
      <td>0.079146</td>
      <td>-0.994820</td>
      <td>-0.113385</td>
      <td>-0.148128</td>
      <td>...</td>
      <td>-0.510770</td>
      <td>0.260886</td>
      <td>-0.133253</td>
      <td>0.175804</td>
      <td>0.030907</td>
      <td>0.005434</td>
      <td>9.552459e-04</td>
      <td>0.632038</td>
      <td>0.399472</td>
      <td>0.252481</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.246336</td>
      <td>1.304613</td>
      <td>-0.012560</td>
      <td>-0.329000</td>
      <td>-0.808132</td>
      <td>-0.981739</td>
      <td>-1.105931</td>
      <td>-0.994820</td>
      <td>0.081045</td>
      <td>0.003094</td>
      <td>...</td>
      <td>0.672141</td>
      <td>0.451774</td>
      <td>0.303656</td>
      <td>0.245060</td>
      <td>0.060055</td>
      <td>0.014717</td>
      <td>3.606557e-03</td>
      <td>-0.831723</td>
      <td>0.691762</td>
      <td>-0.575354</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.378617</td>
      <td>1.304613</td>
      <td>-0.339045</td>
      <td>-0.609036</td>
      <td>-0.808132</td>
      <td>-0.981739</td>
      <td>-1.134602</td>
      <td>0.588023</td>
      <td>0.230591</td>
      <td>0.128368</td>
      <td>...</td>
      <td>0.672141</td>
      <td>0.451774</td>
      <td>0.303656</td>
      <td>-0.222636</td>
      <td>0.049567</td>
      <td>-0.011035</td>
      <td>2.456852e-03</td>
      <td>-0.831723</td>
      <td>0.691762</td>
      <td>-0.575354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.010898</td>
      <td>1.304613</td>
      <td>-2.531499</td>
      <td>-1.315922</td>
      <td>0.550523</td>
      <td>0.250689</td>
      <td>-2.281450</td>
      <td>-0.994820</td>
      <td>0.014341</td>
      <td>0.027589</td>
      <td>...</td>
      <td>-0.860799</td>
      <td>0.740974</td>
      <td>-0.637829</td>
      <td>0.010842</td>
      <td>0.000118</td>
      <td>0.000001</td>
      <td>1.381725e-08</td>
      <td>-1.563603</td>
      <td>2.444854</td>
      <td>-3.822780</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



## Full Model R-Squared

Check out the $R^2$ of the full model with your interaction and polynomial terms added. Print this value for both the train and test data.


```python
# Your code here
```


```python
# __SOLUTION__
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Train R^2:", lr.score(X_train, y_train))
print("Test R^2: ", lr.score(X_test, y_test))
```

    Train R^2: 0.8571817758242435
    Test R^2:  0.6442143449157876


It looks like we may be overfitting some now. Let's try some feature selection techniques.

## Feature Selection

First, test out `RFE` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)) with several different `n_features_to_select` values. For each value, print out the train and test $R^2$ score and how many features remain.


```python
# Your code here

```


```python
# __SOLUTION__
for n in [5, 10, 15, 20, 25]:
    rfe = RFE(LinearRegression(), n_features_to_select=n)
    X_rfe_train = rfe.fit_transform(X_train, y_train)
    X_rfe_test = rfe.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_rfe_train, y_train)

    print("Train R^2:", lr.score(X_rfe_train, y_train))
    print("Test R^2: ", lr.score(X_rfe_test, y_test))
    print(f"Using {n} out of {X_train.shape[1]} features")
    print()
```

    Train R^2: 0.776039994126505
    Test R^2:  0.6352981725272363
    Using 5 out of 26 features
    
    Train R^2: 0.8191862278324273
    Test R^2:  0.6743476159860743
    Using 10 out of 26 features
    
    Train R^2: 0.8483321237427194
    Test R^2:  0.704013767108713
    Using 15 out of 26 features
    
    Train R^2: 0.8495176468836853
    Test R^2:  0.7169477986870836
    Using 20 out of 26 features
    
    Train R^2: 0.8571732578218183
    Test R^2:  0.6459291693655327
    Using 25 out of 26 features
    


Now test out `Lasso` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)) with several different `alpha` values.


```python
# Your code here

```


```python
# __SOLUTION__
for alpha in [1, 10, 100, 1000, 10000]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)

    print("Train R^2:", lasso.score(X_train, y_train))
    print("Test R^2: ", lasso.score(X_test, y_test))
    print(f"Using {sum(abs(lasso.coef_) < 10**(-10))} out of {X_train.shape[1]} features")
    print("and an alpha of", alpha)
    print()
```

    Train R^2: 0.857153074119144
    Test R^2:  0.6485699116355144
    Using 0 out of 26 features
    and an alpha of 1
    
    Train R^2: 0.8571373079024015
    Test R^2:  0.6480527180183058
    Using 0 out of 26 features
    and an alpha of 10
    
    Train R^2: 0.856958744623801
    Test R^2:  0.6471042867008598
    Using 1 out of 26 features
    and an alpha of 100
    
    Train R^2: 0.8506404012942795
    Test R^2:  0.7222278677869791
    Using 9 out of 26 features
    and an alpha of 1000
    
    Train R^2: 0.7790529223548714
    Test R^2:  0.7939567393897818
    Using 14 out of 26 features
    and an alpha of 10000
    


Compare the results. Which features would you choose to use?


```python
# Your written answer here
```


```python
# __SOLUTION__
"""
For RFE the model with the best test R-Squared was using 20 features

For Lasso the model with the best test R-Squared was using an alpha of 10000

The Lasso result was a bit better so let's go with that and the 14 features
that it selected
"""
```

## Evaluate Final Model on Validation Data

### Data Preparation

At the start of this lab, we created `X_val` and `y_val`. Prepare `X_val` the same way that `X_train` and `X_test` have been prepared. This includes scaling, adding interactions, and adding polynomial terms.


```python
# Your code here

```


```python
# __SOLUTION__

# Scale X_val
X_val_scaled = scaler.transform(X_val)
X_val = pd.DataFrame(X_val_scaled, columns=X.columns)

# Add interactions to X_val
for record in top_7_interactions:
    col1, col2 = record[0]
    new_col_name = col1 + "_" + col2
    X_val[new_col_name] = X_val[col1] * X_val[col2]

# Add polynomials to X_val
for (col, degree, _) in top_polynomials.values:
    poly = PolynomialFeatures(degree, include_bias=False)
    col_transformed_val = pd.DataFrame(poly.fit_transform(X_val[[col]]),
                                        columns=poly.get_feature_names([col]))
    X_val = pd.concat([X_val.drop(col, axis=1), col_transformed_val], axis=1)
```

### Evaluation

Using either `RFE` or `Lasso`, fit a model on the complete train + test set, then find R-Squared and MSE values for the validation set.


```python
# Your code here

```


```python
# __SOLUTION__
final_model = Lasso(alpha=10000)
final_model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

print("R-Squared:", final_model.score(X_val, y_val))
print("MSE:", mean_squared_error(y_val, final_model.predict(X_val)))
```

    R-Squared: 0.7991273457324125
    MSE: 1407175023.3081572


## Level Up Ideas (Optional)

### Create a Lasso Path

From this section, you know that when using `Lasso`, more parameters shrink to zero as your regularization parameter goes up. In scikit-learn there is a function `lasso_path()` which visualizes the shrinkage of the coefficients while $alpha$ changes. Try this out yourself!

https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-coordinate-descent-path-py

### AIC and BIC for Subset Selection

This notebook shows how you can use AIC and BIC purely for feature selection. Try this code out on our Ames housing data!

https://xavierbourretsicotte.github.io/subset_selection.html

## Summary

Congratulations! You now know how to apply concepts of bias-variance tradeoff using extensions to linear models and feature selection.
