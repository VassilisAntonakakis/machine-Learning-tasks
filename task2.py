from calendar import c
import string
import numpy as np
from numpy import float64
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer as SI

def int_mapper(df, target_column):

    df_mod = df
    values = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(values)}

    df_mod[target_column] = df_mod[target_column].replace(map_to_int)
    return df_mod


pd.set_option("display.max_rows", None, "display.max_columns", None)
data = pd.read_csv("task2trainning.csv", delimiter=";")
dfTrain = pd.DataFrame(data)

#dfTrain["alcohol"] = pd.to_numeric(dfTrain["alcohol"], errors = 'coerce')

index = 0
for col in dfTrain:
    if dfTrain[col].isnull().values.any():
        print("NaN value at: ", col, " at index: ", index)
    index += 1

kmeans = KMeans(n_clusters = 10, random_state = 0)
kmeans.fit(dfTrain[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]])

label = kmeans.predict(dfTrain[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]])

#plotting the results:
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(dfTrain[label == i , 0] , dfTrain[label == i , 1] , label = i)

plt.legend()
plt.show()
