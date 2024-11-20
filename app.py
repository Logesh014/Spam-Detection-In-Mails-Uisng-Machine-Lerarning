from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Define the directory where model and vectorizer files are stored
BASE_DIR = 'c:/Users/HP/OneDrive/Desktop/SPAM DETECTION IN MAILS USING ML'

# Load your trained model
model_path = os.path.join(BASE_DIR, 'model.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load your vectorizer
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')
if os.path.exists(vectorizer_path):
    with open(vectorizer_path, 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
else:
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_content = request.form['email']  # Get email content from form
        email_features = vectorizer.transform([email_content])  # Transform content
        prediction = model.predict(email_features)  # Predict using the model
        result = 'Ham mail' if prediction[0] == 1 else 'Spam mail'
        return render_template('index.html', prediction=result)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
