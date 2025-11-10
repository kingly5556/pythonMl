import pandas as pd
import numpy as np

data = {
    "Name":["Alice","Bob",np.nan,"David"],
    "Age":[25,np.nan,30,35],
    "Score":[85,90,np.nan,88],
}
df = pd.DataFrame(data)

print("original dataset \n",df)
#fill column age by mean of this column
df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Score"] = df["Score"].interpolate()

print("Dataset: \n",df)