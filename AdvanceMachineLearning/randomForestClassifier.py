from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV

#Load dataset 
data = load_breast_cancer()
x,y = data.data,data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)

print("feature",data.feature_names)
print("Classes",data.target_names)

#train random forest
rf_model = RandomForestClassifier(random_state=50)
rf_model.fit(x_train,y_train)

y_pred = rf_model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print("Random forest score",accuracy)
print("\n Classification report \n",classification_report(y_test,y_pred))

param_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20],
    'max_features':['sqrt','log2',None]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs = -1
)
grid_search.fit(x_train,y_train)

#Display best parameters and score
print("Best parameters",grid_search.best_params_)
print("best crossValidation Accuracy:",grid_search.best_score_)