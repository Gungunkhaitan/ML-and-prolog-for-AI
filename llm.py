import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

np.random.seed(42)
data = {
    "Model": ["LLM", "SLLM", "RAG"] * 50,
    "Accuracy": np.concatenate([
        norm.rvs(loc=0.85, scale=0.05, size=50),  # LLM
        norm.rvs(loc=0.75, scale=0.05, size=50),  # SLLM
        norm.rvs(loc=0.80, scale=0.05, size=50)   # RAG
    ]),
    "F1 Score": np.concatenate([
        norm.rvs(loc=0.83, scale=0.05, size=50),
        norm.rvs(loc=0.72, scale=0.05, size=50),
        norm.rvs(loc=0.78, scale=0.05, size=50)
    ])}
df = pd.DataFrame(data)
plt.figure(figsize=(10, 5))
sns.boxplot(x="Model", y="Accuracy", data=df, palette="coolwarm")
plt.title("Accuracy Comparison of LLM, SLLM, and RAG Models")
plt.show()
plt.figure(figsize=(10, 5))
sns.boxplot(x="Model", y="F1 Score", data=df, palette="coolwarm")
plt.title("F1 Score Comparison of LLM, SLLM, and RAG Models")
plt.show()
X = np.random.rand(150, 2)  # Random features
y = np.array([0] * 75 + [1] * 75)  # Binary labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression F1 Score:", f1_score(y_test, y_pred))
