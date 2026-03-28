import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# get features as dataframe
dataset = load_diabetes(as_frame=True, scaled=False)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df = df.drop(columns='sex') #remove categoical data
df['progression'] = dataset.target

# determine what features to test
f1 = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'] # all possible features
f2 = ['s1', 's2', 's3', 's4', 's5', 's6'] # only serum measurements
f3 = ['age', 's1', 's2', 's3', 's4', 's5', 's6'] # age and serum measurements
f4 = ['bmi', 's1', 's2', 's3', 's4', 's5', 's6'] # bmi and serum measurements
f5 = ['bp', 's1', 's2', 's3', 's4', 's5', 's6'] # bp and serum measurements

# method to create model with given variables
def get_mlr_model(x_vars):
    y = df['progression']
    X = df[x_vars]
    X = sm.add_constant(X)
    model = sm.OLS(y,X).fit()
    return model

# get summaries of models trained on different features
print("A model with blood serum measurements:\n", get_mlr_model(f2).summary())
print("A model with age and serum measurements:\n", get_mlr_model(f3).summary())
print("A model with bmi and serum measurements:\n", get_mlr_model(f4).summary())
print("A model with bp and serum measurements:\n", get_mlr_model(f5).summary())
print("A model with all possible features:'\n", get_mlr_model(f1).summary())
