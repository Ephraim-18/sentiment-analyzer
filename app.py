from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = 0

    if request.method == "POST":
        text = request.form["text"]

        vector = vectorizer.transform([text])
        prediction = model.predict(vector)
        probs = model.predict_proba(vector)

        confidence = round(max(probs[0]) * 100, 2)

        if prediction[0] == 1:
            result = "Positive"
        elif prediction[0] == 0:
            result = "Negative"
        else:
            result = "Neutral"

    return render_template("index.html", result=result, confidence=confidence)


if __name__ == "__main__":
    app.run()
