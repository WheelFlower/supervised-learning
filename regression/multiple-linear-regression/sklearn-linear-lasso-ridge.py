import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

# get features as dataframe
dataset = load_diabetes(as_frame=True, scaled=False)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df = df.drop(columns='sex') #remove categorical data

# scale data
scaler = StandardScaler()
dfcols = df.columns

df_scaled = scaler.fit_transform(df.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=dfcols)

# append target to feature data
df_scaled['progression'] = dataset.target

# split into train-test data
X = df_scaled.drop(columns=['progression'])
y = df_scaled['progression']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#set up models
mlr_model = linear_model.LinearRegression()
ridge_model = linear_model.Ridge(alpha = 0.01)
lasso_model = linear_model.Lasso(alpha = 0.01)

# method to display coefficients
def display_coefficients(model):
    for idx,col_name in enumerate(X_train.columns):
        print("The coefficient for {} is {}".format(col_name, model.coef_[idx]))
    print("The intercept is ", model.intercept_)

# method for displaying accuracy
def display_accuracy(model):
    print('Accuracy for training dataset:', model.score(X_train, y_train))
    print('Accuracy for testing dataset:', model.score(X_test, y_test))

# show coefficients and accuracies for models
print("Linear model coefficients:")
display_coefficients(mlr_model)
print("\nRidge model coefficients:")
display_coefficients(ridge_model)
print("\nLasso model coefficients:")
display_coefficients(lasso_model)

print("\n\nLinear model accuracy:")
display_accuracy(mlr_model)
print("\nRidge model accuracy:")
display_accuracy(ridge_model)
print("\nLasso model accuracy:")
display_accuracy(lasso_model)