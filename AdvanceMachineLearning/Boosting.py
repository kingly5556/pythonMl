from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

data= load_breast_cancer()
x,y = data.data,data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)

# print(data.feature_names,data.target_names)

#Train Gradient boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(x_train,y_train)

y_pred_gb= gb_model.predict(x_test)

#Evaluate performance
accuracy = accuracy_score(y_test,y_pred_gb)
print("Gradient B Accuracy",accuracy)
print("\nClassification report\n",classification_report(y_test,y_pred_gb))

param_grid = {
    'learning_rate':[0.01,0.1,0.2],
    'n_estimators':[50,100,200],
    'max_depth':[3,5,7]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(x_train,y_train)

#Display best parameters and score
print("Best Parameters",grid_search.best_params_)
print("Score:",grid_search.best_score_)

#train random forest
rf_model = RandomForestClassifier(random_state=50)
rf_model.fit(x_train,y_train)

y_pred_rf = rf_model.predict(x_test)

accuracyRF = accuracy_score(y_test,y_pred_rf)

print("Random F Acc",accuracyRF)