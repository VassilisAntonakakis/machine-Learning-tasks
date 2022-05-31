import pandas as pd
import numpy as np

pd.set_option(display.max_rows, None, display.max_columns, None)
data = pd.read_csv(task2trainning.csv, delimiter=;)
dfTrain = pd.DataFrame(data)

dfTrain[alcohol] = pd.to_numeric(dfTrain[alcohol], errors = 'coerce')

for col in dfTrain:
    if dfTrain[col].isnull().values.any():
        print(NaN value at: , col)

for item, frame in dfTrain['alcohol'].iteritems():
    if pd.notnull(frame) == False:
        print (item, frame)