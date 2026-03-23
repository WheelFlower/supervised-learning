import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# get dataset as dataframe
dataset = load_diabetes(as_frame=True, scaled=False)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['progression'] = dataset.target

# remove categorical data and clean
df = df.drop(columns='sex')
df.dropna()

# determine what features to test
f1 = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'] # all possible features
f2 = ['s1', 's2', 's3', 's4', 's5', 's6'] # only serum measurements
f3 =  ['age', 's1', 's2', 's3', 's4', 's5', 's6'] # age and serum measurements
f4 =  ['bmi', 's1', 's2', 's3', 's4', 's5', 's6'] # bmi and serum measurements
f5 =  ['bp', 's1', 's2', 's3', 's4', 's5', 's6'] # bp and serum measurements

# create method to test correlation between features and target