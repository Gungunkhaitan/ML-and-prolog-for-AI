import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('Social_Network_Ads.csv')

X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Decision Tree Classifier with max_depth
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtc.fit(X_train, y_train)

# Predict
y_pred = dtc.predict(X_test)

# Required outputs
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the decision tree (cleaner now)
plt.figure(figsize=(8, 5))
plot_tree(dtc, filled=True, feature_names=['Age', 'EstimatedSalary'], class_names=['No Click', 'Click'])
plt.title("Decision Tree with max_depth=3")
plt.show()


