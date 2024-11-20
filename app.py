from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Define the base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your trained model
model_path = os.path.join(BASE_DIR, 'model.pkl')
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {model_path}")
except Exception as e:
    raise Exception(f"An error occurred while loading the model: {e}")

# Load your vectorizer
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')
try:
    with open(vectorizer_path, 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
except FileNotFoundError:
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
except Exception as e:
    raise Exception(f"An error occurred while loading the vectorizer: {e}")

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
        return render_template('index.html', error=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
