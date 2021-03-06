#Data Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

import pandas as pd
from sklearn import tree as tr
import matplotlib.pyplot as plt

def int_mapper(df, target_column): #function to turn all non numeric values into unique integers

    df_mod = df
    values = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(values)}

    df_mod[target_column] = df_mod[target_column].replace(map_to_int)
    return df_mod

pd.set_option("display.max_rows", None, "display.max_columns", None) #pandas setting to display the entire dataframe
#importing the trainning data
data = pd.read_csv("task3trainning.csv", delimiter=";", dtype="str")
df = pd.DataFrame(data)
df = int_mapper(df, "diagnosis")
df = df.astype('float64')
dfbackup = df.copy

#creating the attributes and classification variables for the decision tree
X, y = df[["radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst"
        ]],df[["diagnosis"]]

#Initializing the decision tree with entropy as the preferred criterion for the nodes
Dtree = tr.DecisionTreeClassifier(criterion="entropy")

#plotting the resulting tree
features = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
classes = ['Malignent', 'Benign']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=200)
tr.plot_tree(Dtree.fit(X, y),
             feature_names=features,
             class_names=classes,
             filled=True)

#importing the evaluation data
del df, dfbackup, data
data = pd.read_csv("task3eval.csv", delimiter=";", dtype="str")
df = pd.DataFrame(data)
df = df.astype('float64')

#predicting brest cancer results for evalation entries
results = Dtree.predict(df[["radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst"
        ]])

#printing the evaluation results
index = 1
for result in results:
    if result == 0:
        print("Result for entry ", index, " is: Malignant Tumor")
    else:
        print("Result for entry ", index, " is: Benign Tumor")
    index += 1

plt.show()