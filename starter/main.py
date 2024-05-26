# Put the code for your API here.
import joblib
import logging
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI

from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load the model
model = joblib.load("model/model.pkl")
lb = joblib.load("model/lb.pkl")
encoder = joblib.load("model/encoder.pkl")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Define the request body


class DataIn(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')

# Define the response body


class DataOut(BaseModel):
    prediction: str

# Define the prediction function


def make_prediction(data: DataIn):
    data = data.dict(by_alias=True)
    # data = pd.DataFrame(data, index=[0])
    data = pd.DataFrame([data])
    # data = data.dropna()

    X_pred, _, _, _ = process_data(
        data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )
    prediction = inference(model, X_pred)
    print(prediction)
    return prediction

# Define the API endpoint


@app.post("/predict", response_model=DataOut)
def predict(data: DataIn):
    y = make_prediction(data)
    prediction = lb.inverse_transform(y)
    return {"prediction": prediction[0]}


@app.get("/")
def read_root():
    return {"Hello": "World"}
