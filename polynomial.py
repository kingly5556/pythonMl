import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Generate Synthetic data
np.random.seed(42)
x = np.random.rand(100,1)*10
y = 3 * x**2+2*x+np.random.randn(100,1)*5

#Transform features to polynomial
poly_features = PolynomialFeatures(degree=2,include_bias=False)
x_poly = poly_features.fit_transform(x)

# Fit Polynomial 
model = LinearRegression()
model.fit(x_poly,y)
y_pred = model.predict(x_poly)

plt.scatter(x,y)
plt.scatter(x,y_pred,colorizer="blue")
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x_poly,y,test_size=0.2,random_state=42)

#ridge regression
ridge_model = Ridge(alpha=1)
ridge_model.fit(x_train,y_train)
ridge_predictions =ridge_model.predict(x_test)

#lasso regress
lasso_model = Lasso(alpha=1)
lasso_model.fit(x_train,y_train)
lasso_predictions = lasso_model.predict(x_test)

#evaluate 
ridge_mse = mean_squared_error(y_test,ridge_predictions)
print("RidgeMSE",ridge_mse)

lasso_mse = mean_squared_error(y_test,lasso_predictions)
print("LassoMSE",lasso_mse)

poly_mse = mean_squared_error(y,y_pred)
print("Polymse",poly_mse)