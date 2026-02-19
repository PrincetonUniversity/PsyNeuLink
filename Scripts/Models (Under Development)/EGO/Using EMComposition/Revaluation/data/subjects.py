import pandas as pd

df = pd.read_csv('twostep.csv')

print(len(pd.unique(df['sub'])))