import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# obtain dataset
iris = load_iris()

# create dataframe
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['species'] = iris.target

# clean potential na
df.dropna()

print(df)
# split train and test data
X = df.drop(columns=['species'])
y = df.species

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# see inital accuracy
dectree = tree.DecisionTreeClassifier()
dectree.fit(X_train,y_train)
y_predicts = dectree.predict(X_test)
accuracy = accuracy_score(y_test,y_predicts)
print('Accuracy:', accuracy)

# show confusion matrix
print('Confusion matrix:')
print(confusion_matrix(y_test,y_predicts))

fig = plt.figure(figsize=(79,90))
tree.plot_tree(dectree, feature_names=X.columns.tolist(), fontsize=20)
plt.show()
fig.savefig('decisiontree1.png')

# predict new values
def predict(values):
    arr = np.array([values])
    new_iris = pd.DataFrame(arr, columns = iris.feature_names)
    prediction = dectree.predict(new_iris)
    if prediction == 0:
        species = "setosa"
    elif prediction == 1:
        species = "versicolor"
    else:
        species = "virginica"
    return species

iris1 = [5.6, 3.2, 4.1, 0.9]
prediction1 = predict(iris1)
print('Example iris:')
print(iris1)
print('Predicted to be:', prediction1)