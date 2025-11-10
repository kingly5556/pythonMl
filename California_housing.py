import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#load the california housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

#Select feature (Median Incoome) and target (Median House Value)
x= df[['MedInc']]
y = df[['MedHouseVal']]

#Transform feature to polynomial feature
poly = PolynomialFeatures(degree=2,include_bias=False)
X_poly = poly.fit_transform(x)
#split data
x_train,x_test,y_train,y_test = train_test_split(X_poly,y,test_size=0.2,random_state=42)


model = LinearRegression()
model.fit(X_poly,y)

y_pred = model.predict(X_poly)

# plt.scatter(x,y,colorizer="blue")
# plt.scatter(x,y_pred,colorizer="res")
# plt.xlabel("Median income")
# plt.ylabel("Median house value")
# plt.show()



#ridge regression
ridge_model = Ridge(alpha=1)
ridge_model.fit(x_train,y_train)
ridge_predictions = ridge_model.predict(x_test)

#lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train,y_train)
lasso_pred=lasso_model.predict(x_test)

#evaluate
polymse = mean_squared_error(y,y_pred)
ridgemse= mean_squared_error(y_test,ridge_predictions)
lassomse= mean_squared_error(y_test,lasso_pred)

print("PolyMSE",polymse,"\n RidgeMSE",ridgemse,"\nLassoMSE",lassomse)

plt.scatter(x_test[:,0],y_test,colorizer="blue")
plt.scatter(x_test[:,0],ridge_predictions,colorizer="red")
plt.scatter(x_test[:,0],lasso_pred,colorizer="green")
plt.xlabel("Median income")
plt.ylabel("Median house value")
plt.show()