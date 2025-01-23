git remote add origin https://github.com/Sumanth-0717/TechPranee-assignment.gitMachine Downtime Prediction API

This project provides a RESTful API for predicting machine downtime using a logistic regression model. The API allows you to upload data, train the model, and make predictions on new input data. The API supports the following endpoints:

/upload - Upload a CSV file containing machine data.
/train - Train the machine learning model using the uploaded data.
/predict - Make predictions on new data.
Requirements

Python 3.x
Required Libraries:
Flask
Pandas
Scikit-learn
Pickle

Data File:
Ensure you have a dataset in CSV format containing the following columns:
Machine_ID
Temperature
Run_Time
Downtime_Flag (1 for downtime, 0 for no downtime)

Setting Up the Application

Step 1: Clone the Repository
Clone the repository or download the source files (e.g., app.py, final_data.csv).

git clone <repository_url>

Step 2: Run the Flask Application
To start the Flask server, run the following command in the terminal:
    -python app.py
This will start the API server on http://127.0.0.1:5000/.

API Endpoints
1. Upload Data
This endpoint allows you to upload a CSV file containing machine data.
Endpoint: POST /upload
Request: Upload a CSV file containing the data.
    -curl -X POST -F "file=@final_data.csv" http://127.0.0.1:5000/upload
Response:
{
  "message": "Data uploaded successfully",
  "columns": ["Machine_ID", "Temperature", "Downtime_Flag", "Run_Time"]
}

2. Train Model
This endpoint trains the logistic regression model using the uploaded data.
Endpoint: POST /train
Request: Trigger training by sending a POST request without data.
    -curl -X POST http://127.0.0.1:5000/train
Response:
{
  "message": "Model trained successfully",
  "accuracy": 0.85,
  "f1_score": 0.82
}

3. Make Prediction
This endpoint makes predictions on new data. The input should include Temperature and Run_Time.
Endpoint: POST /predict
Request: Send a JSON object with Temperature and Run_Time.
    -curl -X POST -H "Content-Type: application/json" -d '{"Temperature": 80, "Run_Time": 120}' http://127.0.0.1:5000/predict
Response:
{
  "Downtime": "Yes",
  "Confidence": 0.85
}

only curl command after running the app.py
1. curl -X POST -F "file=@final_data.csv" http://127.0.0.1:5000/upload
2. curl -X POST http://127.0.0.1:5000/train
3. curl -X POST -H "Content-Type: application/json" -d '{"Temperature": 80, "Run_Time": 120}' http://127.0.0.1:5000/predict
4. curl -X POST -H "Content-Type: application/json" -d '{"Temperature":4.5, "Run_Time": 148}' http://127.0.0.1:5000/predict







