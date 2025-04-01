import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('emails.csv')

X = df['text']
y = df['spam']

user_input = input("Enter your email text: ")

# Vectorization
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.25, random_state=0)

# Train the model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# === Take input from the user ===


# Vectorize user input using the SAME vectorizer
user_vector = vectorizer.transform([user_input])

# Predict
prediction = nb.predict(user_vector)[0]

print("\nPrediction Result:")
if prediction == 1:
    print("The email is classified as: SPAM")
else:
    print("The email is classified as: NOT SPAM")


