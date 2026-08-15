import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("network_traffic_data.csv")

print(df.head())
print(df.describe())
print(df.info())

# Pairplot
sns.pairplot(df, hue="Label")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Distribution plots
for col in df.columns[:-1]:
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()