import pandas as pd
import matplotlib.pyplot as plt

data ={
    "Class":["A","B","A","B","C","C"],
    "Score" :[85,90,88,72,95,80],
    "Age":[15,16,15,17,16,15]
}
df = pd.DataFrame(data)

print("Original Dataset \n",df)
#groupby is method that can group column which you select and find mean and other score like max or min for that group in other column
grouped = df.groupby("Class").mean()
print(grouped)

stats = df.groupby("Class").agg(
    {"Score": ["mean","max","min"],"Age":["mean","max","min"]}
)
print(stats)

plt.bar(data["Age"],data["Score"])
plt.xlabel("Age")
plt.ylabel("Score")
plt.show()