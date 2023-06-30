from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.impute import SimpleImputer

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return "Hello hum 7 sath hai"


@app.route('/predict', methods=['POST'])
def predict():
    task_size = request.form.get('Task_size')
    pages = request.form.get('Pages')
    task_size_local = request.form.get('Task_size_local')
    task_size_rem = request.form.get('Task_size_rem')
    distance = request.form.get('Distance')
    com = request.form.get('Comtime')

    input_query = np.array([[task_size, pages, task_size_local, task_size_rem, distance, com]])

    # Preprocess the data to handle missing values
    imputer = SimpleImputer(strategy='mean')
    input_query = imputer.fit_transform(input_query)

    result = model.predict(input_query)[0]

    return jsonify({'total time': int(result)})  # Convert result to integer


if __name__ == '__main__':
    app.run(debug=True)
