from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression

#Load Iris dataset
data = load_iris()
X,y = data.data,data.target

#split dataset 
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Logistic regress
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(x_train,y_train)

y_pred_lr = log_reg.predict(x_test)

accuracy_lr = accuracy_score(y_test,y_pred_lr)

print("\nlogistic report\n",accuracy_lr)

#knn
best_k = 5
knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(x_train,y_train)
y_pred_knn= knn.predict(x_test)
accuracy_knn = accuracy_score(y_test,y_pred_knn)
print(f"Knn k={best_k}",accuracy_knn)

# #Experiment with different values of k
# for k in range(1,11):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train,y_train)
    
#     y_pred = knn.predict(x_test)
    
#     accuracy = accuracy_score(y_test,y_pred)
#     print(f"k = {k} = {accuracy:.2f}")
