"""Data Ingestion from MongoDB and splitting data into train & test then export them to the Artifacts as csv"""

import os
import sys

import pandas as pd
import pymongo
from pymongo.server_api import ServerApi
from sklearn.model_selection import train_test_split

from src import constants as training_pipeline
from src.config.config_entity import DataIngestionConfig
from src.config.artifacts_entity import DataIngestionArtifact
from utils.exceptions.exceptions import CustomerChurnPrediction
from utils.logging.logger import logging

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Fallback path when MongoDB returns no data (e.g. empty collection)
RAW_DATA_DIR = "data"
RAW_DATA_FILE = "raw"
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE, training_pipeline.DATA_SET_FILE)


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomerChurnPrediction(e, sys)

    def _load_from_mongodb(self) -> pd.DataFrame | None:
        """Read from MongoDB; return DataFrame or None if no data."""
        if not MONGO_DB_URL:
            logging.warning("MONGO_DB_URL not set; skipping MongoDB.")
            return None
        try:
            client = pymongo.MongoClient(MONGO_DB_URL, server_api=ServerApi("1"))
            db = client[self.data_ingestion_config.database_name]
            collection = db[self.data_ingestion_config.collection_name]
            docs = list(collection.find({}))
            if not docs:
                logging.warning("MongoDB collection is empty.")
                return None
            df = pd.DataFrame(docs)
            if df.columns.empty or len(df) == 0:
                return None
            return df
        except Exception as e:
            logging.warning("MongoDB read failed: %s", e)
            return None

    def _load_from_csv_fallback(self) -> pd.DataFrame:
        """Load from raw CSV when MongoDB has no data."""
        path = RAW_CSV_PATH
        if not os.path.isfile(path):
            raise CustomerChurnPrediction(
                ValueError(
                    f"No data from MongoDB and fallback file not found: {path}. "
                    "Set MONGO_DB_URL and ensure the collection has data, or add data/raw/telco_customers.csv"
                ),
                sys,
            )
        logging.info("Loading data from CSV fallback: %s", path)
        return pd.read_csv(path)

    def _clean_and_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop IDs, encode Gender and Churn; same for MongoDB or CSV."""
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        gender_col = "Gender" if "Gender" in df.columns else "gender"
        if gender_col in df.columns:
            df[gender_col] = df[gender_col].replace({"Male": 1, "Female": 0})
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
        return df

    def import_collection_as_dataframe(self) -> pd.DataFrame:
        """Read data from MongoDB, or from raw CSV if MongoDB returns no data."""
        try:
            df = self._load_from_mongodb()
            if df is None or df.columns.empty or len(df) == 0:
                df = self._load_from_csv_fallback()
            df = self._clean_and_encode(df)
            if df.columns.empty or len(df) == 0:
                raise CustomerChurnPrediction(
                    ValueError("No columns or rows after loading and cleaning data."),
                    sys,
                )
            logging.info("Imported dataframe: %s rows, %s columns", len(df), len(df.columns))
            return df
        except CustomerChurnPrediction:
            raise
        except Exception as e:
            raise CustomerChurnPrediction(e, sys)
    def export_data_to_feature_store(self, df: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            df.to_csv(feature_store_file_path, index=False, header=True)
            return df
        except Exception as e:
            raise CustomerChurnPrediction(e, sys)
    def split_data_to_train_test(self, df: pd.DataFrame):
        try:
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path.")

            train_df.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_df.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info("Exported train and test file path.")
        except Exception as e:
            raise CustomerChurnPrediction(e, sys)
    def initiate_data_ingestion(self):
        try:
            df = self.import_collection_as_dataframe()
            df = self.export_data_to_feature_store(df)
            self.split_data_to_train_test(df)

            return DataIngestionArtifact(self.data_ingestion_config.training_file_path, self.data_ingestion_config.testing_file_path)

        except Exception as e:
            raise CustomerChurnPrediction(e, sys)