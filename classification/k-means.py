import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

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
