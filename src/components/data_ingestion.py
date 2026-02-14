"""Data Ingestion form MongoDB and splitting data into train & test then export them to the Artifacts as csv"""

import pymongo
import pandas as pd
import sys
import os
from src.config.config_entity import DataIngestionConfig
from src.config.artifacts_entity import DataIngestionArtifact
from utils.exceptions.exceptions import NetworkSecurityException
from utils.logging.logger import logging
from sklearn.model_selection import train_test_split

MONGO_DB_URL = os.getenv('MONGO_DB_URL')

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def import_collection_as_dataframe(self):
        """
            Read Data from MongoDB
        """
        try:
            mongo_clinet = pymongo.MongoClient(MONGO_DB_URL)
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            collection = mongo_clinet[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if 'customerID' in df.columns.to_list():
                df = df.drop('customerID', axis=1)
            df['Gender'] = df['Gender'].replace({'Male': 1, "Female": 0})
            df['Churn'] = df['Churn'].replace({'Yes': 1, "No": 0})

            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def export_data_to_feature_store(self, df: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            pd.to_csv(feature_store_file_path, index=False, header=True)
            return pd
        except Exception as e:
            raise NetworkSecurityException(e, sys)
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
            raise NetworkSecurityException(e, sys)
    def initiate_data_ingestion(self):
        try:
            df = self.import_collection_as_dataframe()
            df = self.export_data_to_feature_store()
            self.split_data_to_train_test(df)

            return DataIngestionArtifact(self.data_ingestion_config.training_file_path, self.data_ingestion_config.testing_file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys)