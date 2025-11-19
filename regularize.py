from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier

california = fetch_california_housing()
x,y = california.data,california.target
feature_names = california.feature_names

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

lr_model = LinearRegression()
lr_model.fit(x_train,y_train)

y_pred = lr_model.predict(x_test)
mse_lr = mean_squared_error(y_test,y_pred)

print("Linear Regression MSE (NO Regularization):",mse_lr)
print("With Coef:",lr_model.coef_)

#train ridge regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train,y_train)

#predict and evaluate
y_pred_ridge = ridge_model.predict(x_test)
mse_ridge = mean_squared_error(y_test,y_pred_ridge)

print("Ridge Regression MSE:",mse_ridge)
print("Coef",ridge_model.coef_)


