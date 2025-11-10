import pandas as pd

#load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# print(df.head())


#separate feature 
categorical_feature = df.select_dtypes(include=["object"]).columns
numberical_feature = df.select_dtypes(include = ["int64","float64"]).columns

print("\nCategorical feature\n",categorical_feature.tolist)
print("\nNumerical Features:",numberical_feature.tolist)

#Display summary of categorical fearures
print("\n Categorical Feature Summary:\n")
for col in categorical_feature:
    print(f"{col}:\n",df[col].value_counts(),"\n")
    
print("\n Numerical Feature summary:\n")
print(df[numberical_feature].describe())