from calendar import c
import string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def int_mapper(df, target_column):

    df_mod = df
    values = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(values)}

    df_mod[target_column] = df_mod[target_column].replace(map_to_int)
    return df_mod


pd.set_option("display.max_rows", None, "display.max_columns", None)
data = pd.read_csv("task2trainning.csv", delimiter=";")
dfTrain = pd.DataFrame(data)
print(dfTrain.tail())

kmeans = KMeans(n_clusters = 10, random_state = 0).fit(dfTrain[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]])

pd.plotting.scatter_matrix(dfTrain)
plt.show()
