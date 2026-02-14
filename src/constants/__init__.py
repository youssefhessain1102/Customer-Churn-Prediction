import os

import numpy as np

# Common Constant variables for pipeline
TARGET_COLUMN: str = "Churn"

# Feature columns for transformation (used if not inferred from dtypes)
CATEGORICAL_FEATURES: list = [
    "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERICAL_FEATURES: list = [
    "Gender", "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
]
PIPELINE_NAME: str = "CustomerChurnPrediction"
ARTIFACTS_DIR: str = "Artifacts"
DATA_SET_FILE: str = "telco_customers.csv"
TRAINING_FILE: str = "train.csv"
TESTING_FILE: str = "test.csv"
FILE_NAME: str = "telco_customers.csv"

# Data Ingestion Constants
DATA_INGESTION_COLLECTION_NAME: str = "CustomerData"
DATA_INGESTION_DATABASE_NAME: str = "CustomerChurnPrediction"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data Transformation Constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

# Model Trainer Constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05

SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = 'model.pkl'