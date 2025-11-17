from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = load_iris()
x= pd.DataFrame(data.data,columns=data.feature_names)
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Knn classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
print("Accuracy Without Scaling:",accuracy_score(y_test,y_pred))

#Apply min-max scaling
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x_trainS,x_testS,y_trainS,y_testS = train_test_split(x_scaled,y,test_size=0.2,random_state=42)

#train knn classifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(x_trainS,y_trainS)

#predict and evaluate
y_pred_scaled = knn_scaled.predict(x_testS)
print("\n",accuracy_score(y_testS,y_pred_scaled))