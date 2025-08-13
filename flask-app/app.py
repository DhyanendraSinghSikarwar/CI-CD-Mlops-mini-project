from flask import Flask, render_template, request
from preprocessing_utility import normalize_text
import mlflow
import dagshub
import pickle

mlflow.set_tracking_uri('https://dagshub.com/dhyanendra.manit/mlops-mini-project.mlflow')
dagshub.init(repo_owner='dhyanendra.manit', repo_name='mlops-mini-project', mlflow=True)


app = Flask(__name__)

# load model from model registry
model_name = "my_model"
model_version = 3
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Load vectorizer
vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Cleaning
    normalized_text = normalize_text(text)

    # Bow
    vectorized_text = vectorizer.transform([normalized_text])

    # prediction
    result = model.predict(vectorized_text)
    print("Prediction result:", type(result[0]))

    return render_template('index.html', result=str(result[0]))

if __name__ == '__main__':
    app.run(debug=True)
