"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This Application provides a complete serving solution for the telco customer churn
with both programmatic API access and user-friendly web interface.

Architecture:
- FasstAPI: Hight-performance REST API with automatic OpenAPI documentation
- Gradio: User Friendly web UI for manual testing and demonstrations
- Pydantic: Data Validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel  # fix: correct import + correct class name
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)

#=== Health Check EndPoint===
# CRITICAL: Required for AWS application load balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

#=== Request Data Schema===
# Pydantic model model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
    Customer Data schema for churn prediction.
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """
    # Demographics
    gender: str          # "Male" or "Female"
    Partner: str         # "Yes" or "No" - has partner
    Dependents: str      # "Yes" or "No"  - has dependents

    #Phone services
    PhoneService: str    # "Yes" or "No"
    MultipleLines: str   # "Yes", "No" or "No phone service"

    #Internet services
    InternetService: str  # "DSL", "Fiber optic" or "No"
    OnlineSecurity: str   # "Yes", "No" or "No internet service"
    OnlineBackup: str     # "Yes", "No" or "No internet service"
    DeviceProtection: str # "Yes", "No" or "No internet service"
    TechSupport: str      # "Yes", "No" or "No internet service"
    StreamingTV: str      # "Yes", "No" or "No internet service"
    StreamingMovies: str  # "Yes", "No" or "No internet service"

    #Account information
    Contract: str         # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str # "Yes" or "No"
    PaymentMethod: str    # "Electronic check", "Mailed check", etc.

    #Numeric features
    tenure: int           # Number of months with company
    MonthlyCharges: float # Montly charges in dollars
    TotalCharges: float   # Total charges to date

#=== Main Prediction API ENDPOINT===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.

    This endpoint:
    1.Receives validated customer data via Pydantic model
    2.Calls the inference pipeline to transform features and predict
    3.Returns churn prediction in JSON format

    Expected Response:
    - {"prediction" : "Likely to churn"} or {"prediction" : " Not Likely to churn"}
    - {"error": "error_message"} if prediction fails

    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}

#=========================================================================#
# === GRADIO WEB INTERFACE===
def gradio_interface(
        gender, Partner, Dependents, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies, Contract,
        PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    """
    Gradio interface wrapper for the same prediction pipeline used by FastAPI.
    """
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
        result = predict(payload)
        return result
    except Exception as e:
        return f"error: {str(e)}"

# Build the Gradio UI
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
            label="PaymentMethod"
        ),

        gr.Number(label="tenure", value=9),
        gr.Number(label="MonthlyCharges", value=70.0),
        gr.Number(label="TotalCharges", value=70.0),
    ],
    outputs=gr.Textbox(label="prediction"),
    title="Telco Customer Churn - Gradio Demo",
    description="Use this UI to test the same inference pipeline served by FastAPI."
)

# Mount Gradio app under /gradio (so FastAPI serves both API + UI)
# NOTE: Requires recent Gradio versions
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
