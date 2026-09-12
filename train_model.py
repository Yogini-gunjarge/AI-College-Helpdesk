import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("tickets.csv")

# Input and output
X = data["ticket"]
y = data["category"]

# Convert text into numbers
vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)

# Train AI model
model = LogisticRegression()

model.fit(X_vector, y)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("AI Model trained successfully!")
print("Model saved as model.pkl")