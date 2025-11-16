from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data = fetch_california_housing()
x,y = data.data,data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20)
 
model = LinearRegression()
model.fit(x_train,y_train)
ypred = model.predict(x_test)

mae = mean_absolute_error(y_test,ypred)
mse = mean_squared_error(y_test,ypred)
r2 = r2_score(y_test,ypred)

print(mae,"\n",mse,"\n",r2)