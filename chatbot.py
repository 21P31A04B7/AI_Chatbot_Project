# chatbot.py

# Step 1: Import necessary libraries
import random
import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 2: Download NLTK resources
nltk.download('punkt')

# Step 3: Load the intents file
with open('intents.json') as file:
    data = json.load(file)

# Step 4: Prepare training data
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Step 5: Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Step 6: Train the model
model = MultinomialNB()
model.fit(X, tags)

# Step 7: Create a chatbot response function
def chatbot_response(user_input):
    user_input_vector = vectorizer.transform([user_input])
    predicted_tag = model.predict(user_input_vector)[0]

    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm not sure I understand. Can you rephrase?"

# Step 8: Chat loop
print("Chatbot: Hello! I am your AI assistant. Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye! Have a nice day.")
        break
    print("Chatbot:", chatbot_response(user_input))