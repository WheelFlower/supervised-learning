# supervised-learning
Shows different supervised machine learning algorithms, specifically both classification and regression. The diabetes database from sklearn is used for the regression programs and the iris database from sklearn is used in the classification algorithm. The following requirements are fulfilled by this repository:

For all models:
- Transform dataset into a Pandas DataFrame including both the features and the target, keeping column headers.
- Clean data as needed. This includes:
    - Dropping or replacing na values
    - Resolving categorical data
- Split dataset into training and test data
- Scale the feature data using the StandardScaler()
- Display accuracy of the model after the training phase
- Use the model to predict outcomes from new data

Regression (for all models):
- Justify the selection of features for the model by determining the relationship between the feature(s) and the target variable
- Identify the equation created and used by the model to predict values. This includes the slope(s) and y-intercept. (Example: y = mx + b)
In linear regression
- Create a linear regression model using the sklearn library
- Create a linear regression model using the statsmodels library
In multiple linear regression
- Create a multiple linear regression model using the sklearn library
- Create a Lasso regression model using the sklearn library
- Create a Ridge regression model using the skleran library
