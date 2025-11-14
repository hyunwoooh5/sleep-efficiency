from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from predict import Customer, predict_single


class PredictResponse(BaseModel):
    sleep_efficiency: float


app = FastAPI(title="sleep-efficiency")


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer)

    return PredictResponse(
        sleep_efficiency=prob
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
