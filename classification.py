import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
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
y = (x[:,0]*1,5+x[:,1]>15).astype(int)

#create dataframe
df = pd.DataFrame(x,columns=['Age','Salary'])
df['Purchase'] = y

x_train,x_test,y_train,y_test = train_test_split(df[['Age','Salary']],df['Purchase'],test_size=0.2,random_state=42)
