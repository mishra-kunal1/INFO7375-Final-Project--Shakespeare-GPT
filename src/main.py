from flask import Flask, request, jsonify
from inference import generate_text
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data as JSON
    print(data)
    model_input = data['input']
    print(model_input)
    prediction = generate_text(model_input)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))