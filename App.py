from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load model
with open('randomfor.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return jsonify({
        "message": "Student Performance Predictor API",
        "status": "running",
        "endpoints": {
            "predict": "/predict (POST)"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting Student Performance Predictor API...")
    print("📡 API will be available at: http://localhost:5000")
    print("🌐 React frontend should run at: http://localhost:3000")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
