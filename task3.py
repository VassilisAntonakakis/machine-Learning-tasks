import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import matplotlib.pyplot as plt

def int_mapper(df, target_column):

    df_mod = df
    values = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(values)}

    df_mod[target_column] = df_mod[target_column].replace(map_to_int)
    return df_mod

pd.set_option("display.max_rows", None, "display.max_columns", None)
data = pd.read_csv("task3trainning.csv")
df = pd.DataFrame(data)
dfbackup = df.copy

x, y = df[["radius_mean"]].to_numpy(), df[["texture_mean"]].to_numpy


xTrain = x[:-20]
xTest = x[-20:]

yTrain = y
yTest = y

regr = linear_model.LinearRegression()

regr.fit(xTrain, yTrain)

predict = regr.predict(xTest)

print("Predicted values: \n", predict)
print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(yTest, predict))
print("Coefficient of determination: %.2f" % r2_score(yTest, predict))
plt.scatter(xTest, yTest, color="black")
plt.plot(xTest, predict, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()