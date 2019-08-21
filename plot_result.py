import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("record.csv")
print(df.head())

ax=sns.barplot(x="Cleaned",y="mAP",hue="depth",data=df)
plt.show()
