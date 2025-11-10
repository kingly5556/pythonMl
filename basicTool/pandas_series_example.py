#https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
import pandas as pd

#Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

#Explore structure
# print("First 5 rows:\n",df.head())

selected_columns = df[["species","sepal_length"]]
print("Selected Columns: \n",selected_columns)

filtered_rows = df[(df["sepal_length"] > 5.0) & (df['species']=="setosa")]
print("filteres Rows: \n",filtered_rows)

#fill something to column which na
df["column_name"] = df["column_name"].fillna(0)

df.fillna(method="ffill")
df.fillna(method="bfill")

#use for make sure that column will not empty and the value of that column will be some value
df["column_name"] = df['column_name'].interpolate()

#re column name
df.rename(column={"old":"new"})

#change datatype of column
df["column_name"] = df ["column_name"].astype("float")