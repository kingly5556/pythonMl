import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)


print(df.info())
print(df['Class'].value_counts())

x= df.drop(columns=["Class"])
y= df["Class"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#Init Kfold
kf = KFold(n_splits=5,shuffle=True)

rf_model = RandomForestClassifier()
scores_kfold = cross_val_score(rf_model,x_train,y_train,cv=kf,scoring='accuracy')

print("K-Fold CrossValScore:",scores_kfold)
print("Mean Accuracy :",scores_kfold.mean())

#Init K-Fold
skf = StratifiedKFold(n_splits=5,suffle=True)

#train and evaluate model
scores_stratified = cross_val_score(rf_model,x_train,y_train,cv=skf,scoring='accuracy')

print("Stratified K-Fold cross validation score:",scores_stratified)
print("Mean accuracy (KFold):",scores_stratified.mean())