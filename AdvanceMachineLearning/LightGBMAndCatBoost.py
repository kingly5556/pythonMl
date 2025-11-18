import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#Select features and target
feature = ['PassengerId','Sex','Age','Fare','Embarked','Pclass']
target = 'Survived'

#Handle missing values
df.fillna({'Age':df['Age'].median()},inplace = True)
df.fillna({'Embarked':df['Embarked'].mode()[0]},inplace=True)

#Encode categorical variables
label_encoders = {}
for col in ['Sex','Embarked']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#Split data
x = df[feature]
y = df[target]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print("Training data shape:",x_train.shape) 

#training lightGBM
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(x_train,y_train)

#Train catboots
cat_futures=['Pclass','Sex','Embarked']
cat_model = CatBoostClassifier(cat_features=cat_futures,verbose = 0)
cat_model.fit(x_train,y_train)

#predict and evaluate
lgb_pred = lgb_model.predict(x_test)
cat_pred = cat_model.predict(x_test)
print("Light",accuracy_score(y_test,lgb_pred))
print("Cat",accuracy_score(y_test,cat_pred))