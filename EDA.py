import pandas as pd
from dataset import init_datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("dataset.csv")
df["Class"] = df["Class"] - 1
selected_columns = ['V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'Class']
correlation_matrix = df.corr()[selected_columns]

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12})
plt.title('Correlation Heatmap')
plt.show()

count = df.iloc[:, -7:].sum()
count.plot(kind='bar', color='skyblue')
plt.title('Class Counts')
plt.xlabel('Columns')
plt.ylabel('Count')
plt.show()


importance_arr = []

pbar = tqdm(range(1, 2001))

for i in pbar:
    train, test = init_datasets(binary=True, EDA=True)

    x_train, x_test, y_train, y_test = train.x, test.x, train.y, test.y


    # Create a decision tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)

    # Fit the model to your data
    tree_classifier.fit(x_train, y_train)

    feature_importances = tree_classifier.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    top_features = feature_importance_df[feature_importance_df["Importance"] >= 0.05]['Feature'].tolist()

    importance_arr = np.concatenate((importance_arr, top_features))


value_counts = Counter(importance_arr)
values, counts = zip(*value_counts.items())
data = {"Value": values, "Count": counts}
df = pd.DataFrame(data)
df = df.sort_values(by="Count", ascending=False)
plt.figure(figsize=(10, 6))
sns.lineplot(x="Value", y="Count", data=df)
plt.axvline(x=df.iloc[12]["Value"], color='r', linestyle='--', label='Index 3')
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("How important is the feature")
plt.show()
print(df.head(13)['Value'].values)