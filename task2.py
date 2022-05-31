from calendar import c
import string
import numpy as np
from numpy import delete, float64
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
data = pd.read_csv("task2trainning.csv", delimiter=";", dtype='str')
dfTrain = pd.DataFrame(data)
dfTrain = dfTrain.astype('float64')
del data
data = pd.read_csv("task2eval.csv", delimiter=";", dtype='str')
dfEval = pd.DataFrame(data)
dfEval = dfEval.astype('float64')

kmeans = KMeans(n_clusters = 10, random_state = 0)
kmeans.fit(dfTrain[["fixed acidity",
"volatile acidity",
"citric acid",
"residual sugar",
"chlorides",
"free sulfur dioxide",
"total sulfur dioxide",
"density",
"pH",
"sulphates",
"alcohol"]])

labelEval = kmeans.predict(dfTrain[["fixed acidity",
"volatile acidity",
"citric acid",
"residual sugar",
"chlorides",
"free sulfur dioxide",
"total sulfur dioxide",
"density",
"pH",
"sulphates",
"alcohol"]])

rating = {}
index = 0
'''for row in dfTrain.itertuples(index=False):
    for column in row:
        rating[index] += (column * (1 / 11))
        #print(Value: , column)
    index += 1
print(rating)'''



#plotting the results:
u_labels = np.unique(labelEval)
centroids = kmeans.cluster_centers_

for i in u_labels:
    plt.scatter(
    dfTrain[["rating"]],
    dfTrain[["quality"]],
    c = labelEval,
    s = 50,
    cmap = 'viridis'
    )

plt.legend()
plt.show()

