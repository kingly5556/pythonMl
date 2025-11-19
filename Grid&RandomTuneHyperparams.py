from sklearn.datasets import load_iris
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np

data= load_iris()
x,y = data.data ,data.target

#split dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


#define parameter
param_grid =  {
    'n_estimators':[50,100,150],
    'learning_rate':[0.01,0.1,0.2],
    'max_depth':[3,5,7]
}

#initiallize gridSearchCV
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs = -1
)

#Perform grid search
grid_search.fit(x_train,y_train)

best_param_grid =grid_search.best_params_
best_score_grid = grid_search.best_score_
print("Best param grid",best_param_grid)
print("Best cross-validation Acc grid",best_score_grid)

#get best model
best_grid_model = grid_search.best_estimator_

#predict and evaluate
y_pred_grid = best_grid_model.predict(x_test)
accuracy_grid = accuracy_score(y_test,y_pred_grid)

print("Test Acc grid:",accuracy_grid)
print("\n Classification Report\n",classification_report(y_test,y_pred_grid))

#Define parameter distribution
param_dist = {
    'C':np.logspace(-3,3,10),
    'kernel':['linear','rbf','poly','sigmoid'],
    'gamma':['scale','auto']
}

#Initialize RandomizedSearchCv
random_search= RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    n_jobs = -1,
)

random_search.fit(x_train,y_train)

best_param_random = random_search.best_params_
best_score_random = random_search.best_score_

print("Best parameters Randomsearch:",best_param_random)
print("Best cross_val Acc randomsearch:",best_score_random)
