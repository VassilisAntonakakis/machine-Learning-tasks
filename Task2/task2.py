#Wines dataset: https://archive.ics.uci.edu/ml/datasets/wine+quality

import numpy as np
from numpy import delete, float64
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#parsing the trainning and evaluation data into dataframes
pd.set_option("display.max_rows", None, "display.max_columns", None)
data = pd.read_csv("task2trainning.csv", delimiter=";", dtype='str')
dfTrain = pd.DataFrame(data)
dfTrain = dfTrain.astype('float64')
del data
data = pd.read_csv("task2eval.csv", delimiter=";", dtype='str')
dfEval = pd.DataFrame(data)
dfEval = dfEval.astype('float64')

#Setting up a 5 centroid clustering system
kmeans = KMeans(n_clusters = 5, random_state = 0) 
#trainning the clusters
kmeans.fit(dfEval[["fixed acidity",
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


clusterLabel = kmeans.predict(dfTrain[["fixed acidity",
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

#plotting the results:
u_labels = np.unique(clusterLabel)
centroids = kmeans.cluster_centers_

for i in u_labels:
    plt.scatter(
    dfTrain[["rating"]],
    dfTrain[["quality"]],
    c = clusterLabel,
    s = 50
    )


plt.ylabel("Quality")
plt.xlabel("Rating")
plt.title("Πρόβλεψη ποιότητας κόκκινου κρασιού")

plt.show()