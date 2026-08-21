from fastapi import FastAPI,UploadFile,File,HTTPException
import joblib
from pydantic import BaseModel,Field
from typing import Literal
import pandas as pd

app=FastAPI()

model=joblib.load("model.pkl")



class InsuranceInput(BaseModel):
    age:int=Field(...,ge=18,le=100)
    sex:Literal["male","female"]
    bmi:float=Field(...,ge=10,le=60)
    children:int=Field(...,ge=0,le=10)
    smoker:Literal['yes','no']
    region:Literal["southeast","southwest",
                   "northeast","northwest"]


@app.get("/")
def home():
    return{"message":"ML Prediction API is Running"}

@app.post("/predict")
def predict(data:InsuranceInput):
    input_data=pd.DataFrame([data.model_dump()])
    prediction=model.predict(input_data)

    return{
        "predicted_charges":float(prediction[0])
    }

@app.post("/batch-predict")
async def batch_predict(file:UploadFile=File(...)):
    try:
#check file extension
        if not file.filename.endswith(".csv"):
            raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )
#read csv
        df = pd.read_csv(file.file)
#required_columns
        required_columns = [
            "age",
            "sex",
            "bmi",
            "children",
            "smoker",
            "region"
        ]

#find missing columns
        missing_columns=[i for i in required_columns 
                         if i not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns:{missing_columns}")
        
        
#select only ML input columns
        input_df=df[required_columns]

#make predictions:
        predictions=model.predict(input_df)

#add predictions to original dataframe
        df["predictions"]=predictions
##return result 
        return df.to_dict(orient="records")

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    