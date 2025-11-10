from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt


data = load_iris()
x,y =data.data,data.target

# #initialize classifier
# model = RandomForestClassifier(random_state=42)

# #K-fold cross-validation
# kf = KFold(n_splits=5,shuffle=True,random_state=42)
# cv_scores=cross_val_score(model,x,y,cv=kf,scoring="accuracy")

# #Output result
# print("Cross val score:",cv_scores,"\n",cv_scores.mean())

#load dataset
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

#train logistic
model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

#generate the confusion
cm = confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=data.target_names)
disp.plot(cmap="summer")
plt.show()

print("\n classification report\n",classification_report(y_test,y_pred))