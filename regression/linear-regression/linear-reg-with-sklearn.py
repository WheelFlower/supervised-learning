import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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
# isolate X and y
y = df['progression']
X = df[['bmi']]

# split using 80%-20% train and test data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# scale dataset
scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create model
model = LinearRegression()

# train model using training data
model = model.fit(X_train, y_train)

# get equation for the model
y_int = model.intercept_
slope = model.coef_

print('\nThe equation for the model is:')
print('y = bmi *', slope, '+', y_int)

# show model accuracy
#first using the score method
print()
print('Model accuracy:', model.score(X_test,y_test))

# create function to predict progression based on bmi
def predict_from_bmi(bmi):
    bmi = [[bmi]]
    bmi_scaled = scaler.transform(bmi)
    prediction = model.predict(bmi_scaled)
    return prediction

#example prediction (ignore warnings)
print('Prediction for a bmi of 23.3 is', predict_from_bmi(24.1))
