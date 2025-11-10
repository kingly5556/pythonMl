#https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# years = [2010,2011,2012,2013]
# sales = [100,120,140,160]
# # plt.plot(years,sales,label="sales Trend",color="blue",marker="o")
# # plt.title("Sales over Years")
# # plt.xlabel("Years")
# # plt.ylabel("Sales")
# # plt.legend()


# categories = ["Electronics","Clothing","Groceries"]
# revenue = [250,400,150]
# plt.bar(categories,revenue,color = "green")
# plt.show()

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

del df['species']
#calculation correlation matrix
correlation_matrix = df.corr()

#plot heatmap

sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()