from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna as opt


data = load_breast_cancer()
x,y, = data.data,data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#XGB base line model
baseline_model = XGBClassifier(eval_metric ='logloss',random_state=42)
baseline_model.fit(x_train,y_train)

baseline_pred =  baseline_model.predict(x_test)
baseline_acc = accuracy_score(y_test,baseline_pred)
# print("base",baseline_acc)

#Define Objective function for Optuna
def objective(trail):
    params = {
        'n_estimators':trail.suggest_int('n_estimators',50,500),
        'max_depth':trail.suggest_int('max_depth',3,100),
        'learning_rate':trail.suggest_float('learning_rate',0.01,0.3),
        'subsample':trail.suggest_float('subsample',0.6,1.0),
        'colsample_bytree':trail.suggest_float('colsample_bytree',0.6,1.0),
        'gamma':trail.suggest_float('gamma',0,5),
        'reg_alpha':trail.suggest_float('reg_alpha',0,10),
        'reg_lambda':trail.suggest_float('reg_lambda',0,10)
    }
    
    #Train XGBoost model with suggested params
    model = XGBClassifier(eval_metric='logloss',**params)
    model.fit(x_train,y_train)
    
    #Evaluate model on validation set
    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test,preds)
    return accuracy

#create an optuna study
study = opt.create_study(direction="maximize")
study.optimize(objective,n_trials=100)

# #Best hyperparameters
# print(study.best_params)
# print(study.best_value)

param_grid = {
    'n_estimators':[100,200,300],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1,0.2],
    'subsample':[0.6,0.8,1]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss',random_state=42),
    param_grid = param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1
)
grid_search.fit(x_train,y_train)

# print(grid_search.best_params_)
# print("grid:",grid_search.best_score_)

#Define parameter ditributions
param_dist = {
    'n_estimators':[50,100,200,300,400],
    'max_depth':[3,5,7,9],
    'learning_rate':[0.01,0.05,0.1,0.2],
    'subsample':[0.6,0.7,0.9,0.8,1]
}

#train XGB with random search
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss'),
    param_distributions=param_dist,
    n_iter = 50,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)

random_search.fit(x_train,y_train)

#best parameters and accuracy
# print('Randomsearch Best parameters:',random_search.best_params_)
print("random search best accuracy",random_search.best_score_)
print("base",baseline_acc)
print("grid:",grid_search.best_score_)
print("optuna:")
print(study.best_value)

# random search best accuracy: 0.9758191007319623
# base :0.9473684210526315
# grid: 0.9758045776693388
# optuna:0.9649122807017544