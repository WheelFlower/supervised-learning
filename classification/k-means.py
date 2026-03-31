import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.py as plt

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
plt.show()
