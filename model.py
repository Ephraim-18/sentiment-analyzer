import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "text": [
        # POSITIVE
        "I love this product",
        "This is amazing",
        "I feel great",
        "I am very happy",
        "This is fantastic",
        "I really love this",
        "I like this a lot",
        "This is awesome",
        "I absolutely love this",
        "This is very good",
        # NEGATIVE
        "I hate this",
        "This is terrible",
        "I feel bad",
        "This is horrible",
        "I am sad",
        "I dislike this",
        "This is very bad",
        "I really hate this",
        "This is awful",
        "I do not like this",
        "I absolutely hate this product",
        # NEUTRAL
        "this is okay",
        "it is fine",
        "just average",
        "neither good nor bad",
        "it is normal",
        "nothing special",
        "it is okay product",
    ],
    "label": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,  # positive
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # negative
        2,
        2,
        2,
        2,
        2,
        2,
        2,  # neutral
    ],
}
df = pd.DataFrame(data)

# Convert text → numbers
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
X = vectorizer.fit_transform(df["text"])

print("Words:", vectorizer.get_feature_names_out())
print("Vectors:\n", X.toarray().tolist())


#  Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, df["label"], test_size=0.2, random_state=42
)

#  Train model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained successfully")

# Prediction
user_input = input("Enter a sentence: ")

user_vector = vectorizer.transform([user_input])

prediction = model.predict(user_vector)

print("prediction:", prediction[0])
if prediction[0] == 1:
    print("Positive")
elif prediction[0] == 0:
    print("Negative")
else:
    print("Neutral")
text_lower = user_input.lower()

# Strong rule-based correction
if "hate" in text_lower or "terrible" in text_lower or "awful" in text_lower:
    print("Negative")
elif "love" in text_lower or "amazing" in text_lower or "awesome" in text_lower:
    print("Positive")
else:
    if prediction[0] == 1:
        print("Positive")
    elif prediction[0] == 0:
        print("Negative")
    else:
        print("Neutral")


import joblib  # save the model

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully")
