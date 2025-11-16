from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
data = load_iris()
x= data.data
y= (data.target == 0).astype(int)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Not class 0","Class 0"])
disp.plot(cmap="Blues")
plt.show()