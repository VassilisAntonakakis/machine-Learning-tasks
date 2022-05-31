import pandas as pd
from sklearn import tree as tr
import matplotlib.pyplot as plt


def int_mapper(df, target_column):

    df_mod = df
    values = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(values)}

    df_mod[target_column] = df_mod[target_column].replace(map_to_int)
    return df_mod


pd.set_option("display.max_rows", None, "display.max_columns", None)
data = pd.read_csv("task1trainning.csv")
df = pd.DataFrame(data)
df2 = df.copy()

print("Trainning dataframe: \n")
for col in df:
    print(col)
    df2 = int_mapper(df2, col)

print(df2)

X, y = df2[["age, income, housing, married"]], df2[["classifies"]]

# decision node selection criterion setting entropy/gini
Dtree = tr.DecisionTreeClassifier(criterion="gini")

features = ['age', 'income', 'housing', 'married']
classes = ['yes', 'no']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=200)
tr.plot_tree(Dtree.fit(X, y),
             feature_names=features,
             class_names=classes,
             filled=True)

del df, df2, data
data = pd.read_csv("task1eval.csv")
df = pd.DataFrame(data)
dfEval = df.copy()

print("New df: ", dfEval)

for col in df:
    print(col)
    dfEval = int_mapper(dfEval, col)

print("Eval dataframe: \n", dfEval)
results = Dtree.predict(dfEval)

index = 1
for result in results:
    if result == 0:
        print("Result for entry ", index, " is: No")
    else:
        print("Result for entry ", index, " is: Yes")
    index += 1

plt.show()
print()