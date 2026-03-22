import pandas as pd
from sklearn.datasets import load_diabetes
import statsmodels.api as sm

# get dataset as dataframe
dataset = load_diabetes(as_frame=True, scaled=False)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['progression'] = dataset.target

# show first few lines and data info
print('Diabetes dataset:\n', df.head())
print('Dataset info:\n', df.info())

# find correlation between progression (target variable) and other variables
#first get list of x variables to test
x_vars = df.drop(columns='progression')
x_labels = x_vars.columns.to_list()

#then test each variable against progression
for x in x_labels:
    correlation = df[x].corr(df['progression'])
    print('The correlation between', x, ' and progression is:', correlation)

# according to above, bmi has the most correlation to progression. it will be the x variable in the model
# get X and y
y = df['progression']
X = df['bmi']
X = sm.add_constant(X)

# create model
model = sm.OLS(y,X).fit()
print('\nBMI and Performance OLS Model:')
print(model.summary())

# get equation for model
equation = model.params
y_int = equation.iloc[0]
slope = equation.iloc[1]

print('The equation for the model is:')
print('y = x *', slope, '+', y_int)

# use equations to make predictions
def predict_from_bmi(bmi):
    prediction = slope * bmi + y_int
    return prediction

# example prediction
example_bmi = 24.1
print('With a bmi of', example_bmi, 'the predicted progression is', predict_from_bmi(example_bmi))