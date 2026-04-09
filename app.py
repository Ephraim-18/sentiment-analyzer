from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# load the model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        text = request.form["text"]
        print("User input:", text)
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)
        probs = model.predict_proba([text])
        confidence = max(probs[0])
        confidence = round(confidence * 100, 2)
        print("Prediction:", prediction)

        if prediction[0] == 1:
            result = f"Positive   ({confidence}%)"
        elif prediction[0] == 0:
            result = f"Negative  ({confidence}%)"
        else:
            result = f"Neutral  ({confidence}%)"

    return render_template("index.html", result=result, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
