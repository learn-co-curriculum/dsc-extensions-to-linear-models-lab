
# Extensions to Linear Models - Lab

## Introduction

In this lab, you'll practice many concepts you have learned so far, from adding interactions and polynomials to your model to AIC and BIC!

## Summary

You will be able to:
- Build a linear regression model with interactions and polynomial features 
- Use AIC and BIC to select the best value for the regularization parameter 


## Let's get started!

Import all the necessary packages.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import PolynomialFeatures
```

Load the data.


```python
df = pd.read_csv("ames.csv")
```


```python
df = df[['LotArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF',
         '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd',
         'GarageArea', 'Fireplaces', 'SalePrice']]
```

## Look at a baseline housing data model

Above, we imported the Ames housing data and grabbed a subset of the data to use in this analysis.

Next steps:

- Split the data into target (`y`) and predictors (`X`) -- ensure these both are DataFrames 
- Scale all the predictors using `scale`. Convert these scaled features into a DataFrame 
- Build at a baseline model using *scaled variables* as predictors. Use 5-fold cross-validation (set `random_state` to 1) and use the $R^2$ score to evaluate the model 


```python
y = df[['SalePrice']]
X = df.drop(columns='SalePrice')

X_scaled = scale(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

all_data = pd.concat([y, X_scaled], axis=1)
```


```python
regression = LinearRegression()

crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)
baseline = np.mean(cross_val_score(regression, X_scaled, y, scoring='r2', cv=crossvalidation))
baseline
```




    0.7524751004088885



## Include interactions

Look at all the possible combinations of variables for interactions by adding interactions one by one to the baseline model. Next, evaluate that model using 5-fold cross-validation and store the $R^2$ to compare it with the baseline model.

Print the 7 most important interactions.


```python
combinations = list(combinations(X.columns, 2))

interactions = []
data = X_scaled.copy()
for comb in combinations:
    data['interaction'] = data[comb[0]] * data[comb[1]]
    score = np.mean(cross_val_score(regression, data, y, scoring='r2', cv=crossvalidation))
    if score > baseline: interactions.append((comb[0], comb[1], round(score, 3)))
            
print('Top 7 interactions: %s' %sorted(interactions, key=lambda inter: inter[2], reverse=True)[:7])
```

    Top 7 interactions: [('OverallQual', 'TotRmsAbvGrd', 0.77), ('OverallQual', 'GarageArea', 0.764), ('OverallQual', '2ndFlrSF', 0.758), ('2ndFlrSF', 'GrLivArea', 0.756), ('2ndFlrSF', 'TotRmsAbvGrd', 0.756), ('OverallQual', 'Fireplaces', 0.754), ('OverallCond', 'TotalBsmtSF', 0.754)]


Write code to include the 7 most important interactions in your data set by adding 7 columns. Name the columns "var1_var2", where var1 and var2 are the two variables in the interaction.


```python
df_inter = X_scaled.copy()
ls_interactions = sorted(interactions, key=lambda inter: inter[2], reverse=True)[:7]
for inter in ls_interactions:
    df_inter[inter[0] + '_' + inter[1]] = X[inter[0]] * X[inter[1]]
```

## Include polynomials

Try polynomials of degrees 2, 3, and 4 for each variable, in a similar way you did for interactions (by looking at your baseline model and seeing how $R^2$ increases). Do understand that when going for a polynomial of 4, the particular column is raised to the power of 2 and 3 as well in other terms. We only want to include "pure" polynomials, so make sure no interactions are included. We want the result to return a list that contain tuples of the form:

`(var_name, degree, R2)`, so eg. `('OverallQual', 2, 0.781)` 


```python
polynomials = []
for col in X.columns:
    for degree in [2, 3, 4]:
        data = X_scaled.copy()
        poly = PolynomialFeatures(degree, include_bias=False)
        X_transformed = poly.fit_transform(X[[col]])
        data = pd.concat([data.drop(col, axis=1),pd.DataFrame(X_transformed)], axis=1)
        score = np.mean(cross_val_score(regression, data, y, scoring='r2', cv=crossvalidation))
        if score > baseline: polynomials.append((col, degree, round(score, 3)))
print('Top 10 polynomials: %s' %sorted(polynomials, key=lambda poly: poly[2], reverse=True)[:10])
```

    Top 10 polynomials: [('GrLivArea', 4, 0.807), ('GrLivArea', 3, 0.788), ('OverallQual', 2, 0.781), ('OverallQual', 3, 0.779), ('OverallQual', 4, 0.779), ('2ndFlrSF', 3, 0.775), ('2ndFlrSF', 2, 0.771), ('2ndFlrSF', 4, 0.771), ('GarageArea', 4, 0.767), ('GarageArea', 3, 0.758)]


For each variable, print out the maximum $R^2$ possible when including Polynomials.


```python
polynom = pd.DataFrame(polynomials)
polynom.groupby([0], sort=False)[2].max()
```




    0
    OverallQual     0.781
    OverallCond     0.753
    2ndFlrSF        0.775
    GrLivArea       0.807
    TotRmsAbvGrd    0.753
    GarageArea      0.767
    Name: 2, dtype: float64



Which two variables seem to benefit most from adding polynomial terms?

Add Polynomials for the two features that seem to benefit the most, as in have the best $R^2$ compared to the baseline model. For each of the two features, raise to the Polynomial that generates the best result. Make sure to start from the data set `df_inter` so the final data set has both interactions and polynomials in the model.


```python
for col in ['OverallQual', 'GrLivArea']:
    poly = PolynomialFeatures(4, include_bias=False)
    X_transformed = poly.fit_transform(X[[col]])
    colnames= [col, col + '_' + '2',  col + '_' + '3', col + '_' + '4']
    df_inter = pd.concat([df_inter.drop(col, axis=1), pd.DataFrame(X_transformed, columns=colnames)], axis=1)
```

Check out your final data set and make sure that your interaction terms as well as your polynomial terms are included.


```python
df_inter.head()
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
      <th>OverallQual_TotRmsAbvGrd</th>
      <th>OverallQual_GarageArea</th>
      <th>...</th>
      <th>OverallQual_Fireplaces</th>
      <th>OverallCond_TotalBsmtSF</th>
      <th>OverallQual</th>
      <th>OverallQual_2</th>
      <th>OverallQual_3</th>
      <th>OverallQual_4</th>
      <th>GrLivArea</th>
      <th>GrLivArea_2</th>
      <th>GrLivArea_3</th>
      <th>GrLivArea_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>-0.207142</td>
      <td>-0.517200</td>
      <td>-0.459303</td>
      <td>-0.793434</td>
      <td>1.161852</td>
      <td>0.912210</td>
      <td>0.351000</td>
      <td>-0.951226</td>
      <td>56</td>
      <td>3836</td>
      <td>...</td>
      <td>0</td>
      <td>4280</td>
      <td>7.0</td>
      <td>49.0</td>
      <td>343.0</td>
      <td>2401.0</td>
      <td>1710.0</td>
      <td>2924100.0</td>
      <td>5.000211e+09</td>
      <td>8.550361e+12</td>
    </tr>
    <tr>
      <td>1</td>
      <td>-0.091886</td>
      <td>2.179628</td>
      <td>0.466465</td>
      <td>0.257140</td>
      <td>-0.795163</td>
      <td>-0.318683</td>
      <td>-0.060731</td>
      <td>0.600495</td>
      <td>36</td>
      <td>2760</td>
      <td>...</td>
      <td>6</td>
      <td>10096</td>
      <td>6.0</td>
      <td>36.0</td>
      <td>216.0</td>
      <td>1296.0</td>
      <td>1262.0</td>
      <td>1592644.0</td>
      <td>2.009917e+09</td>
      <td>2.536515e+12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.073480</td>
      <td>-0.517200</td>
      <td>-0.313369</td>
      <td>-0.627826</td>
      <td>1.189351</td>
      <td>-0.318683</td>
      <td>0.631726</td>
      <td>0.600495</td>
      <td>42</td>
      <td>4256</td>
      <td>...</td>
      <td>7</td>
      <td>4600</td>
      <td>7.0</td>
      <td>49.0</td>
      <td>343.0</td>
      <td>2401.0</td>
      <td>1786.0</td>
      <td>3189796.0</td>
      <td>5.696976e+09</td>
      <td>1.017480e+13</td>
    </tr>
    <tr>
      <td>3</td>
      <td>-0.096897</td>
      <td>-0.517200</td>
      <td>-0.687324</td>
      <td>-0.521734</td>
      <td>0.937276</td>
      <td>0.296763</td>
      <td>0.790804</td>
      <td>0.600495</td>
      <td>49</td>
      <td>4494</td>
      <td>...</td>
      <td>7</td>
      <td>3780</td>
      <td>7.0</td>
      <td>49.0</td>
      <td>343.0</td>
      <td>2401.0</td>
      <td>1717.0</td>
      <td>2948089.0</td>
      <td>5.061869e+09</td>
      <td>8.691229e+12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.375148</td>
      <td>-0.517200</td>
      <td>0.199680</td>
      <td>-0.045611</td>
      <td>1.617877</td>
      <td>1.527656</td>
      <td>1.698485</td>
      <td>0.600495</td>
      <td>72</td>
      <td>6688</td>
      <td>...</td>
      <td>8</td>
      <td>5725</td>
      <td>8.0</td>
      <td>64.0</td>
      <td>512.0</td>
      <td>4096.0</td>
      <td>2198.0</td>
      <td>4831204.0</td>
      <td>1.061899e+10</td>
      <td>2.334053e+13</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



## Full model R-squared

Check out the $R^2$ of the full model.


```python
full_model = np.mean(cross_val_score(regression, df_inter, y, scoring='r2', cv=crossvalidation))
full_model
```




    0.8245917461916372



## Find the best Lasso regularization parameter

You learned that when using Lasso regularization, your coefficients shrink to 0 when using a higher regularization parameter. Now the question is which value we should choose for the regularization parameter. 

This is where the AIC and BIC come in handy! We'll use both criteria in what follows and perform cross-validation to select an optimal value of the regularization parameter $alpha$ of the Lasso estimator.

Read the page here: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html and create a similar plot as the first one listed on the page. 


```python
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV, LassoLarsIC
```


```python
model_bic = LassoLarsIC(criterion='bic')
model_bic.fit(df_inter, y)
alpha_bic_ = model_bic.alpha_

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(df_inter, y)
alpha_aic_ = model_aic.alpha_


def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color, linewidth=2, label= name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=2,
                label='alpha for %s ' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'green')
plot_ic_criterion(model_bic, 'BIC', 'blue')
plt.legend()
plt.title('Information-criterion for model selection');
```


![png](index_files/index_33_0.png)


## Analyze the final result

Finally, use the best value for the regularization parameter according to AIC and BIC, and compare $R^2$ and RMSE using train-test split. Compare with the baseline model.

Remember, you can find the Root Mean Squared Error (RMSE) by setting `squared=False` inside the function (see [the documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)), and the RMSE returns values that are in the same units as our target - so we can see how far off our predicted sale prices are in dollars.


```python
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
```


```python
# Split X_scaled and y into training and test sets
# Set random_state to 1
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=1)

# Code for baseline model
linreg_all = LinearRegression()
linreg_all.fit(X_train, y_train)

# Print R-Squared and RMSE
print('Training R-Squared:', linreg_all.score(X_train, y_train))
print('Test R-Squared:', linreg_all.score(X_test, y_test))
print('Training RMSE:', mean_squared_error(y_train, linreg_all.predict(X_train), squared=False))
print('Test RMSE:', mean_squared_error(y_test, linreg_all.predict(X_test), squared=False))
```

    Training R-Squared: 0.7478270652928448
    Test R-Squared: 0.8120708166668684
    Training RMSE: 39424.15590381302
    Test RMSE: 35519.17035590487



```python
# Split df_inter and y into training and test sets
# Set random_state to 1
X_train, X_test, y_train, y_test = train_test_split(df_inter, y, random_state=1)

# Code for lasso with alpha from AIC
lasso = Lasso(alpha= model_aic.alpha_) 
lasso.fit(X_train, y_train)

# Print R-Squared and RMSE
print('Training R-Squared:', lasso.score(X_train, y_train))
print('Test R-Squared:', lasso.score(X_test, y_test))
print('Training RMSE:', mean_squared_error(y_train, lasso.predict(X_train), squared=False))
print('Test RMSE:', mean_squared_error(y_test, lasso.predict(X_test), squared=False))
```

    Training R-Squared: 0.8446714993955369
    Test R-Squared: 0.8657420069305382
    Training RMSE: 30941.3132234915
    Test RMSE: 30021.734184476485



```python
# Code for lasso with alpha from BIC
lasso = Lasso(alpha= model_bic.alpha_) 
lasso.fit(X_train, y_train)

# Print R-Squared and RMSE
print('Training R-Squared:', lasso.score(X_train, y_train))
print('Test R-Squared:', lasso.score(X_test, y_test))
print('Training RMSE:', mean_squared_error(y_train, lasso.predict(X_train), squared=False))
print('Test RMSE:', mean_squared_error(y_test, lasso.predict(X_test), squared=False))
```

    Training R-Squared: 0.8446487101363189
    Test R-Squared: 0.8660207515757948
    Training RMSE: 30943.582941357854
    Test RMSE: 29990.55263037502


## Level up (Optional)

### Create a Lasso path

From this section, you know that when using Lasso, more parameters shrink to zero as your regularization parameter goes up. In Scikit-learn there is a function `lasso_path()` which visualizes the shrinkage of the coefficients while $alpha$ changes. Try this out yourself!

https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-coordinate-descent-path-py

### AIC and BIC for subset selection
This notebook shows how you can use AIC and BIC purely for feature selection. Try this code out on our Ames housing data!

https://xavierbourretsicotte.github.io/subset_selection.html

## Summary

Congratulations! You now know how to create better linear models and how to use AIC and BIC for both feature selection and to optimize your regularization parameter when performing Ridge and Lasso. 
