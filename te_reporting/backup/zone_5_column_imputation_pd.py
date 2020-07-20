import pandas as pd
df = pd.read_csv("../imputed_predict_2020_06_30_1.csv")
mean = df.mean()
maximum = df.max()
minimum = df.min()
std = df.std()
median = df.median()
df = pd.concat([mean,maximum,minimum, std, median])
df.to_csv('reports/zone5_imputation.csv')
