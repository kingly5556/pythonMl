import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report,roc_auc_score
df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.info())

print(df['Churn'].value_counts())
print(df.head())

df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()},inplace = True)

label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column]= label_encoder.fit_transform(df[column])
#Encode target variable
df['Churn'] = label_encoder.fit_transform(df['Churn'])

#Scale numerical features
scaler = StandardScaler()
numerical_feature = ['tenure','MonthlyCharges','TotalCharges']
df[numerical_feature] = scaler.fit_transform(df[numerical_feature])

x = df.drop(columns=['Churn'])
y = df['Churn']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#apply SMOTE
smote = SMOTE()
x_train_resampled,y_train_resampled = smote.fit_resample(x_train,y_train)

#Display class distribution after SMOTE
print("\n Class Distribution after SMOTE")
print(pd.Series(y_train_resampled).value_counts())

#Train random forest
rf_model = RandomForestClassifier()
rf_model.fit(x_train_resampled,y_train_resampled)
y_pred_rf = rf_model.predict(x_test)
roc_auc_rf = roc_auc_score(y_test,rf_model.predict_proba(x_test)[:,1])

#Train XGB
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(x_train_resampled,y_train_resampled)
y_pred_xgb = xgb_model.predict(x_test)
roc_auc_xgb = roc_auc_score(y_test,xgb_model.predict_proba(x_test)[:,1])

#LightGBM
lgb_model = LGBMClassifier()
lgb_model.fit(x_train_resampled,y_train_resampled)
y_pred_lgb = lgb_model.predict(x_test)
roc_auc_lgb = roc_auc_score(y_test,lgb_model.predict_proba(x_test)[:,1])

#report
print("RF Report \n",classification_report(y_test,y_pred_rf),roc_auc_rf)
print("XGB Report \n",classification_report(y_test,y_pred_xgb),roc_auc_xgb)
print("LGB Report \n",classification_report(y_test,y_pred_lgb),roc_auc_lgb)