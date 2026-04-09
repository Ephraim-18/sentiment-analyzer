text = input("Enter a sentence: ").lower()

positive_words = ["good", "amazing", "happy", "joyful", "excellent"]
negative_words = ["bad", "terrible", "sad", "horrible", "awful"]

if "not" in text:
    if any(word in text for word in positive_words):
        print("negative")  # not good negaitive
    elif any(word in text for word in negative_words):
        print("positive")  # not bad positive
    else:
        print("neutral")
else:

    if any(word in text for word in positive_words):
        print("positive")
    elif any(word in text for word in negative_words):
        print("negative")
    else:
        print("neutral")
