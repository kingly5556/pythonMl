import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
data = load_breast_cancer()
x,y = data.data,data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)

#convert dataset to Dmatrix
dtrain = xgb.DMatrix(x_train,label = y_train)
dtest = xgb.DMatrix(x_test,label = y_test)

#train
param = {
    'objective':'binary:logistic',
    'eval_metric':'logloss',
    'max_depth':3,
    'eta':0.1
}

xgb_model = xgb.train(param,dtrain,num_boost_round=100)
y_pred = (xgb_model.predict(dtest)>0.5).astype(int)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of XGBOOST",accuracy)
print("REPORT\n",classification_report(y_test,y_pred))

param_grid = {
    'learning_rate':[0.01,0.1,0.2],
    'n_estimators':[50,100,200],
    'max_depth':[3,5,7],
    'subsample':[0.8,1.0],
    'colsample_bytree':[0.8,1.0]
}

#Initialize XGboost classifier
xgb_clf = XGBClassifier(eval_metric = 'logloss')

#perform grid
grid_search = GridSearchCV(estimator=xgb_clf,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1)

grid_search.fit(x_train,y_train)
print("Best param:",grid_search.best_params_)
print("Best Cross-Val Acc:",grid_search.best_score_)

gb_model = GradientBoostingClassifier()
gb_model.fit(x_train,y_train)
y_pred_gb = gb_model.predict(x_test)

accuracy_gb= accuracy_score(y_test,y_pred_gb)

print("gradient Boosting:",accuracy_gb)