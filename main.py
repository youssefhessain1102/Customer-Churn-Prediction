from utils.exceptions.exceptions import CustomerChurnPrediction
import sys
from src.components.data_ingestion import DataIngestion
from src.config.config_entity import DataTransformationConfig, DataIngestionConfig, TrainingPipelineConfig, ModelTrainerConfig
from src.config.artifacts_entity import DataIngestionArtifact, DataTransformationArtifact
from src import constants as training_pipeline
from utils.logging.logger import logging

if __name__ == "__main__":
    try:

        # Data Ingestion
        trainig_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(trainig_pipeline_config)
        data_ingstion = DataIngestion(data_ingestion_config)
        logging.info("Initiate Data ingestion")

        data_ingestion_artifacts = data_ingstion.initiate_data_ingestion()
        logging.info("Ingestion Initiation completed")
    except Exception as e:
        raise CustomerChurnPrediction(e, sys)