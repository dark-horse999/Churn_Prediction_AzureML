# Churn_Prediction_AzureML
# Customer Churn Prediction on Azure ML

This project demonstrates a full **MLOps workflow** on Azure ML for training, deploying, and monitoring a customer churn prediction model.

---

## 🚀 Project Overview
- **Train** a machine learning model on Azure ML using a registered compute cluster and environment.  
- **Register** the trained model in the workspace model registry.  
- **Deploy** the model to an Azure ML online endpoint.  
- **Test** the endpoint with REST API calls.  
- **Monitor** predictions using Application Insights.  

---

## 📂 Repository Structure
├── environment.yml # Conda environment file (dependencies)
├── train-aml.yml # Azure ML training job definition
├── endpoint.yml # Azure ML endpoint configuration
├── predict.py # Scoring script for inference
├── data/ # Training dataset(s)
└── README.md # Project documentation
