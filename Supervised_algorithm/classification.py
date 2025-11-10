import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
import pandas as pd

# def sigmoid(z):
#     return 1/(1 + np.exp(-z))

# #generate value
# z = np.linspace(-10,10,100)
# sigmoid_values = sigmoid(z)

# plt.plot(z,sigmoid_values)
# plt.show() 

#generate synthetic data
np.random.seed(42)
n_samples = 200
x = np.random.rand(n_samples,2)*10
y = (x[:,0]*1.5+x[:,1]>15).astype(int)

#create dataframe
df = pd.DataFrame(x,columns=['Age','Salary'])
df['Purchase'] = y

x_train,x_test,y_train,y_test = train_test_split(df[['Age','Salary']],df['Purchase'],test_size=0.2,random_state=42)

#logistic model
model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Acc:",accuracy_score(y_test,y_pred))
print("Prec:",precision_score(y_test,y_pred))
print("Rec:",recall_score(y_test,y_pred))
print("F1:",f1_score(y_test,y_pred))
print("\nClassification Report\n:",classification_report(y_test,y_pred))

import matplotlib.pyplot as plt

#Plot dicision boundary
x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))

#Predict propabilities for grid points
z = model.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)

plt.contourf(xx,yy,z,alpha=0.8,cmap="coolwarm")
plt.scatter(x_test['Age'],x_test['Salary'],c=y_test,edgecolors="k",cmap="coolwarm")
plt.title("logistic regression decision Boundary")
plt.xlabel("age")
plt.ylabel("Salary")
plt.show()