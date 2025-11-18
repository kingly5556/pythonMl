import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

#load dataset 
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
# print(df["Class"].value_counts())

x = df.drop(columns=['Class'])
y = df['Class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

rf_model = RandomForestClassifier(class_weight="balanced")
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
print(classification_report(y_test,y_pred))

roc_auc = roc_auc_score(y_test,rf_model.predict_proba(x_test)[:,1])
print("ROC-AUC",roc_auc)

#Apply SMOTE
smote = SMOTE(random_state=42)
x_resampled,y_resampled = smote.fit_resample(x_train,y_train)

#Display new class distribution
print("\n class distribution after SMOTE\n")
print(pd.Series(y_resampled).value_counts())

#train random forest on resampled data
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(x_resampled,y_resampled)

y_pred_smote = rf_model_smote.predict(x_test)
print("\n Classification Report SMOTE:\n")

roc_auc_smote = roc_auc_score(y_test,rf_model_smote.predict_proba(x_test)[:,1])

print(classification_report(y_test,y_pred_smote))

print("ROC-AUC SMOTE:",roc_auc_smote)