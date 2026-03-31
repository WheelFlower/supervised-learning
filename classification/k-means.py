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
