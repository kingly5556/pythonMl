import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df =pd.read_csv(url)

#Select relevant feature
df = df[['Pclass','Sex','Age','Fare','Embarked','Survived']]

#Handle missing values
df.fillna({'Age':df['Age'].median()},inplace=True)
df.fillna({'Embarked':df["Embarked"].mode()[0]},inplace=True)

#Define feature and target
x = df.drop(columns=['Survived'])
y = df['Survived']

#apply feature scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),['Age','Fare']),
        ('cat',OneHotEncoder(),['Pclass','Sex','Embarked'])
    ]
)

x_preprocessed = preprocessor.fit_transform(x)

#Train and evakuate Logistic regression
log_model = LogisticRegression()
log_scores = cross_val_score(log_model,x_preprocessed,y,cv=5,scoring='accuracy')
print("Logistic accuracy",log_scores.mean(),"\n")

#train and evaluate random forest
rf_model = RandomForestClassifier(random_state=50)
rf_scores = cross_val_score(rf_model,x_preprocessed,y,cv=5,scoring='accuracy')
print("Random F accuracy",rf_scores.mean())

#Define hyperparameter grid
param_grid = {
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20],
    'min_samples_split':[2,5,10]
}

#perform grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(x_preprocessed,y)

print("Vest hyperparameters:",grid_search.best_params_)
print("Best accuracy:",grid_search.best_score_)