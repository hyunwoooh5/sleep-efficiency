import json
import pickle
from typing import Literal
from pydantic import BaseModel, Field


class Customer(BaseModel):
    age: int = Field(..., ge=0)
    gender: Literal["Male", "Female"]
    sleep_duration: float = Field(..., ge=0.0)
    deep_sleep_percentage: int = Field(..., ge=0)
    light_sleep_percentage: int = Field(..., ge=0)
    awakenings: float = Field(..., ge=0)
    caffeine_consumption: float = Field(..., ge=0)
    alcohol_consumption: float = Field(..., ge=0)
    smoking_status: Literal["Yes", "No"]
    exercise_frequency: float = Field(..., ge=0)
    sleep_time: float = Field(..., ge=0)
    getup_time: float = Field(..., ge=0)


with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_single(customer):
    gender_int = 1 if customer.gender == 'Female' else 0
    smoking_int = 1 if customer.smoking_status == 'Yes' else 0

    features = [
        customer.age,
        gender_int,
        customer.sleep_duration,
        customer.deep_sleep_percentage,
        customer.light_sleep_percentage,
        customer.awakenings,
        customer.caffeine_consumption,
        customer.alcohol_consumption,
        smoking_int,
        customer.exercise_frequency,
        customer.sleep_time,
        customer.getup_time
    ]

    result = model.predict([features])
    return float(result[0])


def lambda_handler(event, context):
    print("Parameters:", event)

    if 'body' in event:
        # Call HTTP (curl)
        data = json.loads(event['body'])
    else:
        data = event

    customer = data['customer']
    customer_model_instance = Customer(**customer)
    prob = predict_single(customer_model_instance)
    return prob
