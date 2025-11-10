#Task 1 :perform EDA and prepocessing
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

#define features and target
x= df[['MedInc','HouseAge','AveRooms']]
y = df[['MedHouseVal']]

# print(df.describe())

# #visualize
# sns.pairplot(df,vars=['MedInc','AveRooms','HouseAge','MedHouseVal'])
# plt.show()

# print("Missing Values: \n",df.isnull().sum())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
print("MSE",mse)