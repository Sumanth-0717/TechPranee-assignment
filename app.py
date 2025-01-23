from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

model = None
data = None

@app.route('/upload', methods=['POST'])
def upload_data():
    global data
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        data = pd.read_csv(file)
        return jsonify({"message": "Data uploaded successfully", "columns": list(data.columns)}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 500


@app.route('/train', methods=['POST'])
def train_model():
    global model, data
    if data is None:
        return jsonify({"error": "No data uploaded"}), 400

    try:
        required_columns = ['Temperature', 'Run_Time', 'Downtime_Flag']
        for col in required_columns:
            if col not in data.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        X = data[['Temperature', 'Run_Time']]
        y = data['Downtime_Flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return jsonify({"message": "Model trained successfully", "accuracy": accuracy, "f1_score": f1}), 200
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            return jsonify({"error": "Model not trained yet"}), 400

    try:
        input_data = request.json
        temperature = input_data.get('Temperature')
        run_time = input_data.get('Run_Time')

        if temperature is None or run_time is None:
            return jsonify({"error": "Missing required inputs: 'Temperature' and 'Run_Time'"}), 400
        input_df = pd.DataFrame([[temperature, run_time]], columns=['Temperature', 'Run_Time'])
        prediction = model.predict(input_df)[0]
        confidence = max(model.predict_proba(input_df)[0])

        return jsonify({"Downtime": "Yes" if prediction == 1 else "No", "Confidence": confidence}), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
