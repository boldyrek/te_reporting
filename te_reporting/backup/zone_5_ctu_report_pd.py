import pandas as pd
from sys import argv
type(argv[1])
file_name = argv[1]
df = pd.read_csv(file_name)
df = df.groupby(['CTU']).agg(['min','max','median']).transpose()
df.to_csv('reports/zone_5_ctu_report.csv')
df = df.reset_index()
df.columns = df.columns.astype(str)
print(df)
df.to_feather('reports/zone_5_ctu_report.feather')


