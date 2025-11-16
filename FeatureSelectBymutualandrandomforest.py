from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
#Load the dataset
data = load_diabetes()
df = pd.DataFrame(data.data,columns=data.feature_names)
df['target'] = data.target

#display dataset information
print(df.head())
print(df.info())

#Correlation matrix
correlation_matrix = df.corr()

#Plot heatmap
# plt.figure(figsize=(10,8))
# sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
# plt.show()

#Select features with hihg correlation to the target
correlated_features = correlation_matrix['target'].sort_values(ascending=False)
# print("Features Most correlated with Target:")
# print(correlated_features)

#separate featured and target 
x = df.drop(columns=['target'])
y = df['target']

#calculate mutual information
mutual_info = mutual_info_regression(x,y)

#Create a dataframe for better visualization
mi_df = pd.DataFrame({'Feature':x.columns,"Mutual Information":mutual_info})
mi_df = mi_df.sort_values(by="Mutual Information",ascending=False)

# print("Mutual Information scores:")
# print(mi_df)

#train a random forest model
model = RandomForestRegressor(random_state=42)
model.fit(x,y)

#Get future importance 
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({"feature":x.columns,'Importance':feature_importance})
importance_df = importance_df.sort_values(by='Importance',ascending=False)

print("feature importance from random forest:")
print(importance_df)