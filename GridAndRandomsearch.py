from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


data = load_iris()
x,y = data.data,data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#Define Hyperparameter grid
param_grid = {
    'n_estimators':[50,100,150],
    'max_depth':[None,5,10],
    'min_samples_split':[2,5,10]
}

#Initialize Grid Search
grid_search = GridSearchCV(
    estimator = RandomForestClassifier(),
    param_grid = param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

#perform grid search
grid_search.fit(x_train,y_train)

#evaluate the best model
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(x_test)
accuracy_grid = accuracy_score(y_test,y_pred_grid)

print("Best Hyperparameters :",grid_search.best_params_)
print(accuracy_grid)

param_dist = {
    'n_estimators':np.arange(50,200,10),
    'max_depth':[None,5,10,15],
    'min_samples_split':[2,5,10,20]
}

#Initialize random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

#perform random search
random_search.fit(x_train, y_train)

#evaluate the best model
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(x_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print("Best Hyperparameters from Random Search :", random_search.best_params_)
print("Random Search Accuracy:", accuracy_random)
