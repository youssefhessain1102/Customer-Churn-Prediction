from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
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

        # Data Transformation
        data_transformation_config = DataTransformationConfig(trainig_pipeline_config)
        data_transformation = DataTransformation(data_ingestion_artifacts, data_transformation_config)
        logging.info("Initiate Data Transformation")

        data_transformation_artifacts = data_transformation.initiate_data_transformation()
        logging.info("Transformation Initiation completed")

        # Model Training
        model_trainer_config = ModelTrainerConfig(trainig_pipeline_config)
        model_trainer = ModelTrainer(data_transformation_artifacts, model_trainer_config)
        logging.info("Initiate Model Training")
        model_trainer_artifact = model_trainer.initiate_model_training()
        logging.info("Model Training completed")
        logging.info("Model Trainer Artifact: %s", model_trainer_artifact)

    except Exception as e:
        raise CustomerChurnPrediction(e, sys)