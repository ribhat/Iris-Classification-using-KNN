import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


iris_df = sns.load_dataset('iris')
#print(iris_df.head())


sns.boxplot(x = 'species', y = 'sepal_length', data = iris_df)

sns.distplot(iris_df['sepal_width'])

sns.countplot(x='petal_width', data = iris_df)

sns.pairplot(iris_df, hue = 'species')


scaler = StandardScaler()
scaler.fit(iris_df.drop('species', axis = 1))
scaled_four_features = scaler.transform(iris_df.drop('species', axis = 1))

iris_df_four_features = pd.DataFrame(scaled_four_features, columns = iris_df.
iris_df_four_features.head()

x = iris_df_four_features
y = iris_df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, ran
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(x_train, y_train)
prediction_one = knn.predict(x_test)
