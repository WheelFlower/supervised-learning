# supervised-learning
Each file in this repository displays a supervised machine learning algorithm. The models are split into two folders: regression and classification. Regression models predict continuous values while classification models predict categories or labels. 
## For all models:
- Transform dataset into a Pandas DataFrame including both the features and the target, keeping column headers.
- Clean data as needed. This includes:
    - Dropping or replacing na values
    - Resolving categorical data
- Display accuracy of the model after the training phase
- Use the model to predict outcomes from new data
## Regression Models
In machine learning, regression models aim to find the best fitting line or curve to describe the relationship between the independent and target variables. This line/curve is then used to predict new values plugged into the model. In this respositary, two main types of regression are represented: linear regression and multiple linear regression.
### Linear regression
Linear regression is the simplest form of regression where the model maps the relationship between one independent variable and the target variable. If graphed, this relationship would be represented with a straight line with the equation y = mx + b where y would be the target variable and x would be the independent variable. Within this repository, there are two linear regression models that use statsmodels and sklearn respectively to show the process involved in creating the same model using said libraries.

Outside of the requirements for every model, the linear regression models also:
- Justify the selection of features for the model by determining the relationship between the feature(s) and the target variable
- Identify the equation created and used by the model to predict values. This includes the slope and y-intercept (example: y = mx + b)

linear-reg-statsmodels.py shows a model trained on the entire raw dataset while linear-reg-sklearn.py details a process for standardizing and splitting the dataset into training and test data. The reason the different processes are shown is because there are situations in which they would be necessary or unnecessary. Some datasets have larger numeric ranges in the feature variables which would require the features to be scaled to ensure all features contribute proportionally to the model, in other datasets this would not be necessary. When it comes to train-test splitting, sometimes the dataset does not need to be split into training and test data if testing data has already been set aside beforehand in a separate file. 
### Multiple linear regression
Multiple linear regression requires the model to map the relationship between multiple independent varaibles and the target variable. 

mlr-basic.py displays a multiple linear regression model achiveved through statsmodels. Outside of the requirements for every model in the repository, this model:
- Uses the StandardScaler from sklearn to standardize the data

There are two popular techniques for regularizing linear regression models: Ridge regression and Lasso regression. Both methods operate by applying a penalty term to restrain the model's coefficients, but they go about it in different ways. Ridge regression adds a penalty based on the squared magnitude of the coefficients while lasso regression adds a penalty based on the absolute values of the coefficients. Ridge regression tends to perform better for the purpose of addressing multicollinearity (when two independent variables are highly correlated) while Lasso regression better addresses the elimination of insignificant independent variables.

To showcase the differences between a model using Ridge regression, Lasso regression, and multiple linear regression, sklearn-linear-lasso-ridge.py portrays each of these models when trained on the same training data. Using the sklearn library, each model:
- Uses a train-test split of 70%-30%
- Uses feature data is scaled using sklearn's StandardScaler
- Displays the coeffients of the features used
- Displays the accuracy score of both test and training data to demonstrate the possibility of overfitting (conforming too closely to the training data)
## Classification Models
In machine learning, classification models operate by learning class characteristics from training data and then using those learned characteristics to assign possible classes to new data. 

The k-means.py file uses the sklearn KNeighborsClassifier model. As well as the requirements for all models in this repository, this model:
- Uses a train-test split of 70%-30%
- Feature data is standardized using sklearn's StandardScaler
- Computes the accuracy scores for the various k-values the model could use and plots them to show which is highest
