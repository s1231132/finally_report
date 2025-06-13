import pandas as pd
df = pd.read_csv('ai4i2020.csv')
print(df)
df.to_csv('output.csv', index=False,encoding='big5')
