import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example data
emails = ["Free money now!!!", "Hi, how are you?", "Congratulations, you won a prize!", "Let's meet tomorrow."]
labels = [0, 1, 0, 1]  # 0 = Spam, 1 = Ham

# Create a vectorizer
vectorizer = CountVectorizer()
email_features = vectorizer.fit_transform(emails)

# Train a simple model
model = MultinomialNB()
model.fit(email_features, labels)

# Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Files created successfully!")
