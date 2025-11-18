from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

data = load_breast_cancer()
x,y= data.data,data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Train random forest
rf_default = RandomForestClassifier()
rf_default.fit(x_train,y_train)

y_pred_default = rf_default.predict(x_test)
accuracy_default = accuracy_score(y_test,y_pred_default)
print("Default model accuracy:",accuracy_default)
print("\n classification report:\n",classification_report(y_test,y_pred_default))

#Train random forest with adjuster hyperparameters
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
rf_tuned.fit(x_train,y_train)

#Predict and evaluate
y_pred_tuned = rf_tuned.predict(x_test)
accuracy_tuned = accuracy_score(y_test,y_pred_tuned)

print("Tuned model:",accuracy_tuned)
print("\n Classification Report:\n",classification_report(y_test,y_pred_tuned))