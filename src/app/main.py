from fastapi import FastAPI, Body
from pydantic import BaseModel, ConfigDict
import gradio as gr

from src.serving.inference import predict  # doit exister dans inference.py

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"status": "ok"}


# ✅ Exemple unique pour Swagger / Try it out
EXAMPLE_PAYLOAD = {
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 12,
    "MonthlyCharges": 85.5,
    "TotalCharges": 1025.6,
}


class CustomerData(BaseModel):
    # ✅ Pydantic v2 (optionnel mais utile)
    model_config = ConfigDict(
        json_schema_extra={"example": EXAMPLE_PAYLOAD}
    )

    gender: str
    Partner: str
    Dependents: str

    PhoneService: str
    MultipleLines: str

    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def get_prediction(
    data: CustomerData = Body(
        ...,
        examples={
            "default": {
                "summary": "Default example",
                "description": "Change only the values, keep the keys.",
                "value": EXAMPLE_PAYLOAD,
            }
        },
    )
):
    try:
        payload = data.model_dump()      # Pydantic v2 ✅
        result = predict(payload)        # predict() retourne un dict
        return result
    except Exception as e:
        return {"error": str(e)}


def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    try:
        payload = {
            "gender": gender,
            "Partner": Partner,
            "Dependents": Dependents,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "tenure": int(tenure),
            "MonthlyCharges": float(MonthlyCharges),
            "TotalCharges": float(TotalCharges),
        }

        return predict(payload)

    except Exception as e:
        return {"error": str(e)}


gradio_app = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="gender"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),

        gr.Dropdown(["Yes", "No"], label="PhoneService"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="MultipleLines"),

        gr.Dropdown(["DSL", "Fiber optic", "No"], label="InternetService"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="OnlineSecurity"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="OnlineBackup"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="DeviceProtection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="TechSupport"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="StreamingTV"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="StreamingMovies"),

        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["Yes", "No"], label="PaperlessBilling"),
        gr.Dropdown(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            label="PaymentMethod",
        ),

        gr.Number(label="tenure", value=12),
        gr.Number(label="MonthlyCharges", value=85.5),
        gr.Number(label="TotalCharges", value=1025.6),
    ],
    outputs=gr.JSON(label="prediction"),
    title="Telco Customer Churn - Gradio Demo",
)

app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
