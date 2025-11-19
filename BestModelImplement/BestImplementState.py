import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import numpy as np

df = pd.read_csv("Telco-Customer-Churn.csv")

#Display data to see about balance
print(df['Churn'].value_counts())

#Handle Missing value
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df.fillna({'TotalCharges':df['TotalCharges'].median()},inplace = True)

#Encode categorical varibles
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = label_encoder.fit_transform(df[column])
        
#Encode target variable
df['Churn'] = label_encoder.fit_transform(df['Churn'])

#Scale numerical features
scaler = StandardScaler()
numerical_features = ['tenure','MonthlyCharges','TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

#Features and Target
x = df.drop(columns=['Churn'])
y= df['Churn']

#split data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#train init model
rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)

#Evaluate initial model
y_pred = rf_model.predict(x_test)
accuracy_initial = accuracy_score(y_test,y_pred)

print("Initial Model Accuracy:" ,accuracy_initial)
print("Classification Report\n",classification_report(y_test,y_pred))

#Define parameter grid
param_dist = {
    'n_estimators': np.arange(50,200,10),
    'max_depth':[None,5,10,15],
    'min_samples_split':[2,5,10,20],
    'min_samples_leaf':[1,2,4]
}

#initialze randomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

#perform randomized search
random_search.fit(x_train,y_train)

#Get best parameters
best_params_randomS = random_search.best_params_
print("Best parameters RandomizedSearchCV:",best_params_randomS)

#train best model
best_model = random_search.best_estimator_

#predict and evaluate
y_pred_tuned = best_model.predict(x_test)
accuracy_tuned = accuracy_score(y_test,y_pred_tuned)

print("Tunes model Accuracy:",accuracy_tuned)
print("\nClassification Report(Tuned Model):\n",classification_report(y_test,y_pred_tuned))

#Evaluate using cross-validation
cv_scores = cross_val_score(best_model,x,y,cv=5,scoring='accuracy')