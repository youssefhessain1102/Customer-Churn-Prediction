"""Pushing dataset to the MongoDB Collection"""
import os
import json
import certifi
from dotenv import load_dotenv
import pandas as pd
import pymongo

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

ca = certifi.where()

class NetworkDataExtract:
    """inserting Dataset to MongoDB"""  
    def __init__(self):
        pass

    def csv_to_json_convertor(self, file_path):
        """Convert CSV file to JSON"""
        data = pd.read_csv(file_path)
        data.reset_index(drop=True, inplace=True)
        _records = list(json.loads(data.T.to_json()).values())
        return _records    
    def insert_data_mongodb(self, records, database, collection_path):
        """inserting Dataset to MongoDB"""  
        self.database = database
        self._collection = collection_path
        self.records = records

        self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
        self.database = self.mongo_client[self.database]

        self._collection = self.database[self._collection]
        self._collection.insert_many(self.records)
        return len(self.records)

if __name__ == "__main__":
    FILE_PATH = "data/raw/telco_customers.csv"
    DATABASE = "CustomerChurnPrediction"
    COLLECTION = "CustomerData"
    networkobj = NetworkDataExtract()
    records_to_mongo = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records_to_mongo)
    no_of_records = networkobj.insert_data_mongodb(records_to_mongo, DATABASE, COLLECTION)
    print(no_of_records)
