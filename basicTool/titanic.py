import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#load data 
url ="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#inspect data
print(df.info())
print(df.describe())

#Handle missing value
df["Age"] = df["Age"].fillna(df["Age"].median())#set missing age by median
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])#set missing Embarked using mode or most frequent of Embarked 

# Remove duplicates
df = df.drop_duplicates()

#fillter data:Passengers in first class
first_class = df[df["Pclass"]==1]
print("First Class Passengers: \n",first_class.head())

#Bar Chart:Survival rate by class
# survival_by_class = df.groupby("Pclass")["Survived"].mean()
# survival_by_class.plot(kind="bar",color="skyblue")
# plt.title("Survival Rate by class")
# plt.ylabel("survival rate")
# plt.show()

#Histogram: Age distribution
# sns.histplot(df["Age"],kde=True,bins=30,color="purple")
# plt.title("Age Distribution")
# plt.xlabel("Age")
# plt.ylabel("frequency")
# plt.show()

#Scatter Plot:Age vs Fare
plt.scatter(df["Age"],df["Fare"],alpha=0.5,color="green")
plt.xlabel("age")
plt.ylabel("fare")
plt.show()

