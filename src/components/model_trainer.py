import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config.artifacts_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.config.config_entity import ModelTrainerConfig
from utils.model_selection import tune_classifier
from utils.exceptions.exceptions import CustomerChurnPrediction
from utils.logging.logger import logging
from src import constants as training_pipeline


class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    @staticmethod
    def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Mirror `DataTransformation._prepare_data` so y aligns with saved X arrays.

        - Coerce numerical columns to numeric
        - Drop rows with any missing values
        """
        for col in training_pipeline.NUMERICAL_FEATURES:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()

    @staticmethod
    def _to_binary_target(y) -> np.ndarray:
        """
        Convert common churn target encodings to 0/1.

        Supports:
        - numeric/boolean already (0/1, True/False)
        - strings like "Yes"/"No", "True"/"False", "1"/"0"
        """
        y_arr = np.asarray(y)
        if y_arr.dtype.kind in {"b", "i", "u"}:
            return y_arr.astype(int)
        if y_arr.dtype.kind == "f":
            return np.nan_to_num(y_arr, nan=0.0).astype(int)

        y_str = pd.Series(y_arr).astype(str).str.strip().str.lower()
        return y_str.isin(["yes", "true", "1"]).astype(int).to_numpy()

    def train_model(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models with hyperparameter tuning and select the best one.
        
        Parameters
        ----------
        X_train : np.ndarray
            Transformed training features
        X_test : np.ndarray
            Transformed test features
        y_train : np.ndarray
            Training target
        y_test : np.ndarray
            Test target
            
        Returns
        -------
        ModelTrainerArtifact
            Contains trained model path and metrics
        """
        try:
            models = {
                "RandomForestClassifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier()
            }

            params = {
                "RandomForestClassifier": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10],
                    "class_weight": ["balanced"]
                },
                "XGBClassifier": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200],
                    "scale_pos_weight": [3, 5]  # Adjust based on your churn ratio
                }
            }

            best_model = None
            best_model_name = None
            best_score = -np.inf
            best_model_info = None

            # Tune each model and find the best one
            for model_name, model in models.items():
                logging.info(f"Tuning {model_name}...")
                param_grid = params.get(model_name, {})
                
                tuned_model, tuning_info = tune_classifier(
                    estimator=model,
                    param_grid=param_grid,
                    x_train=X_train,
                    y_train=y_train,
                    cv=5,
                    scoring="f1",  # Using f1_score for imbalanced classification
                    n_jobs=-1
                )
                
                logging.info(
                    f"{model_name} - Best CV Score: {tuning_info['best_score']:.4f}, "
                    f"Best Params: {tuning_info['best_params']}"
                )
                
                # Track the best model based on CV score
                if tuning_info['best_score'] > best_score:
                    best_score = tuning_info['best_score']
                    best_model = tuned_model
                    best_model_name = model_name
                    best_model_info = tuning_info

            if best_model is None:
                raise CustomerChurnPrediction("No model was successfully trained", sys)

            logging.info(f"Best model: {best_model_name} with CV score: {best_score:.4f}")
            if best_model_info is not None:
                logging.info("Best model params: %s", best_model_info.get("best_params"))

            # Calculate metrics on training set
            y_train_pred = best_model.predict(X_train)
            train_f1 = f1_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)

            # Calculate metrics on test set
            y_test_pred = best_model.predict(X_test)
            test_f1 = f1_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)

            logging.info(
                f"Training Metrics - F1: {train_f1:.4f}, Precision: {train_precision:.4f}, "
                f"Recall: {train_recall:.4f}"
            )
            logging.info(
                f"Test Metrics - F1: {test_f1:.4f}, Precision: {test_precision:.4f}, "
                f"Recall: {test_recall:.4f}"
            )

            # Check if model meets expected accuracy
            if test_f1 < self.model_trainer_config.expected_accuracy:
                logging.warning(
                    f"Model F1 score {test_f1:.4f} is below expected {self.model_trainer_config.expected_accuracy}"
                )

            # Check for overfitting/underfitting
            score_diff = abs(train_f1 - test_f1)
            if score_diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(
                    f"Potential overfitting detected. Train-Test F1 difference: {score_diff:.4f} "
                    f"(threshold: {self.model_trainer_config.overfitting_underfitting_threshold})"
                )

            # Save the best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)
            logging.info(f"Saved best model ({best_model_name}) to {self.model_trainer_config.trained_model_file_path}")

            # Create metric artifacts
            trained_metric_artifact = ClassificationMetricArtifact(
                f1_score=train_f1,
                precision=train_precision,
                recall=train_recall
            )

            tested_metric_artifact = ClassificationMetricArtifact(
                f1_score=test_f1,
                precision=test_precision,
                recall=test_recall
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                trained_metric_artifact=trained_metric_artifact,
                tested_metric_artifact=tested_metric_artifact
            )

        except Exception as e:
            raise CustomerChurnPrediction(e, sys)

    def initiate_model_training(self) -> ModelTrainerArtifact:
        """
        Load transformed data and target variables, then train models.
        
        Returns
        -------
        ModelTrainerArtifact
            Contains trained model path and metrics
        """
        try:
            # Load transformed features
            X_train = np.load(self.data_transformation_artifacts.transformed_train_file_path)
            X_test = np.load(self.data_transformation_artifacts.transformed_test_file_path)

            # Get the directory structure to find original CSV files for target variables
            transformed_dir = os.path.dirname(self.data_transformation_artifacts.transformed_train_file_path)
            artifacts_dir = os.path.dirname(os.path.dirname(transformed_dir))
            data_ingestion_dir = os.path.join(artifacts_dir, training_pipeline.DATA_INGESTION_DIR_NAME)
            ingested_dir = os.path.join(data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR)
            
            train_csv_path = os.path.join(ingested_dir, training_pipeline.TRAINING_FILE)
            test_csv_path = os.path.join(ingested_dir, training_pipeline.TESTING_FILE)

            # Load target from CSV files
            train_df = self._prepare_data(pd.read_csv(train_csv_path))
            test_df = self._prepare_data(pd.read_csv(test_csv_path))

            target = training_pipeline.TARGET_COLUMN
            y_train = self._to_binary_target(train_df[target].values)
            y_test = self._to_binary_target(test_df[target].values)

            if X_train.shape[0] != y_train.shape[0]:
                raise CustomerChurnPrediction(
                    ValueError(
                        f"X_train rows ({X_train.shape[0]}) != y_train rows ({y_train.shape[0]}). "
                        "This usually means the CSV was cleaned/dropped differently than the transformed arrays. "
                        "Re-run the pipeline to regenerate artifacts, or ensure the same cleaning is applied."
                    ),
                    sys,
                )

            if X_test.shape[0] != y_test.shape[0]:
                raise CustomerChurnPrediction(
                    ValueError(
                        f"X_test rows ({X_test.shape[0]}) != y_test rows ({y_test.shape[0]}). "
                        "This usually means the CSV was cleaned/dropped differently than the transformed arrays. "
                        "Re-run the pipeline to regenerate artifacts, or ensure the same cleaning is applied."
                    ),
                    sys,
                )

            logging.info(f"Loaded training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
            logging.info(f"Loaded test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

            # Train models
            return self.train_model(X_train, X_test, y_train, y_test)

        except Exception as e:
            raise CustomerChurnPrediction(e, sys)
