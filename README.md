#FastAPI ML Prediction System

A Machine Learning prediction system built using **Python, FastAPI, Scikit-learn, Pandas, Pydantic, and Uvicorn**.

The project exposes REST APIs for real-time and batch machine learning predictions.

##Features

- Train a Scikit-learn machine learning model
- Save the trained model using Joblib
- Create REST APIs using FastAPI
- Validate API inputs using Pydantic
- Real-time prediction using '/predict'
- Batch prediction using CSV upload
- CSV input validation
- Basic error handling
- Automatic API documentation using Swagger UI

##Technologies

- Python
- FastAPI
- Scikit-learn
- Pandas
- Pydantic
- Uvicorn
- Joblib
- REST API

##Project Structure

```text
FastAPI_ML_Prediction_System/
│
├── main.py              # FastAPI application
├── train_model.py       # Model training
├── model.pkl            # Trained ML model
├── insurance.csv        # Dataset
├── requirements.txt     # Required Python libraries
└── README.md            # Project documentation


## Project Workflow
Dataset
  
Data Preprocessing
   
Model Training
   
Model Evaluation
   
Save model.pkl
   
FastAPI
   
Pydantic Validation
   
Prediction
   
JSON Response


#Future Improvements
Deploy API to cloud
Add authentication
Add logging and monitoring
Improve model performance
Add automated testing
Add CI/CD pipeline

