import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('train.csv')

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values: Yellow bars indicate missing data')
plt.show()

df['Age'] = df['Age'].fillna(df['Age'].mean())

df.drop('Cabin', axis=1, inplace=True)

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, predictions)*100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
