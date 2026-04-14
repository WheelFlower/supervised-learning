import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# obtain dataset
iris = load_iris()

# create dataframe
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['species'] = iris.target

# clean potential na
df.dropna()

#scale feature data
X_raw = df.drop(columns=['species'])
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# split train and test
y = df['species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# compute k-values
k_values = [i for i in range(1,50)]
scores = []
for k in k_values:
  knn = KNeighborsClassifier(n_neighbors=k)
  score = cross_val_score(knn,X,y,cv=5)
  scores.append(np.mean(score))

# graph possible k-values
sns.lineplot(x=k_values,y=scores,marker='o')
plt.xlabel('K-Values')
plt.ylabel('Accuracy Scores')
plt.savefig('k-values.png')
plt.show()

# get max k-value according to accuracy
highest_accuracy = max(scores)
index = scores.index(highest_accuracy)
k_value = k_values[index]

# create model based on optimal k-value
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train,y_train)

# show accuracy score, classification report, and confusion matrix
print('Accuracy Score:')
y_predicts = model.predict(X_test)
print(accuracy_score(y_test,y_predicts))

print('Classification Report:')
print(classification_report(y_test, y_predicts))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_predicts))

# create method to predict new values
def predict(arr, model):
  values = np.array([arr])
  new_iris = pd.DataFrame(values, columns=X_raw.columns)
  new_iris = scaler.transform(new_iris)
  prediction = model.predict(new_iris)
  if prediction == 0:
    species = "setosa"
  elif prediction == 1:
    species = "versicolor"
  else:
    species = "virginica"
  return species
