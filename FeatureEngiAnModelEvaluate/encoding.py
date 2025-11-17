import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
#Apply one hot
df_ont_hot = pd.get_dummies(df,columns=['Sex','Embarked'],drop_first = True)

# print(df_ont_hot.head())

label_encoder = LabelEncoder()
df['Pclass_encoded']= label_encoder.fit_transform(df['Pclass'])

# print("\n Label Encoded\n")
# print(df[['Pclass','Pclass_encoded']].head())

#frequency encoding 
df['Ticket_frequency']= df['Ticket'].map(df['Ticket'].value_counts())

# print(df[['Ticket','Ticket_frequency']].head())

x=df_ont_hot.drop(columns=['Survived','Name','Ticket','Cabin'])
y = df['Survived']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='median')
x_imputed = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

x_train,x_test,y_train,y_test =  train_test_split(x_imputed,y,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Accuracy is",accuracy_score(y_test,y_pred))
