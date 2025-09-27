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

## 🔑 Prerequisites
1. An active [Azure subscription](https://azure.microsoft.com/free/).  
2. Azure CLI with ML extension installed:

Below are the steps in Azure CLI
az login
az account set -s <your-subscription-id>

##Creating Compute resources for Model Training

az ml compute create \
  --name my-aml-cluster \
  --size Standard_DS3_v2 \
  --min-instances 0 \
  --max-instances 4 \
  --type AmlCompute \
  -g <Resource Group> -w <workspace>
  
##Creating Environment  

az ml environment create \
  --name churn-ml-env \
  --version 1 \
  --conda-file environment.yml \
  --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest \
  -g <Resource Group> -w <workspace>

##Register Training Data

  az ml data create \
  --name churn-train \
  --version 1 \
  --path customer_churn_dataset-training-master.csv \
  --type uri_file \
  -g <Resource Group> -w <workspace>

##Submit Training Job

az ml job create --file train-aml.yml -g <Resource Group> -w <workspace>

az ml job stream --name churn-train-job -g <Resource Group> -w <workspace>

##Register Trained Model

az ml model create \
  -n churn_pipeline \
  --path "azureml://jobs/churn-train-job/outputs/model_output/paths/churn_pipeline.joblib" \
  -g <Resource Group> -w <workspace>

##Create & Deploy Online Endpoint

az ml online-endpoint create -f endpoint.yml -g <Resource Group> -w <workspace>

az ml online-deployment create -f deployment-staging.yml -g <RESOURCE_GROUP> -w <WORKSPACE_NAME>

az ml online-endpoint update -n churn-endpoint --traffic "blue=100" -g <RESOURCE_GROUP> -w <WORKSPACE_NAME>


##Check Endpoint Details & Get Authentication Keys

az ml online-endpoint show -n churn-endpoint -g <RESOURCE_GROUP> -w <WORKSPACE_NAME> --query scoring_uri -o tsv

az ml online-endpoint get-credentials -n churn-endpoint -g <RESOURCE_GROUP> -w <WORKSPACE_NAME>

##Test the Endpoint

$headers = @{
  "Authorization" = "Bearer <your-primary-or-secondary-key>"
  "Content-Type"  = "application/json"
}

Invoke-RestMethod `
  -Uri "https://<your-endpoint>.<region>.inference.ml.azure.com/score" `
  -Method POST `
  -Headers $headers `
  -Body '[{"Age":41,"Gender":"Female","Tenure":28,"Usage Frequency":28,"Support Calls":7,"Payment Delay":13,"Subscription Type":"Standard","Contract Length":"Monthly","Total Spend":584,"Last Interaction":20}]'







