"""
Data Transformation: build a single preprocessor (OHE + StandardScaler),
fit on train data, save as pkl, export transformed train/test arrays as npy,
and return DataTransformationArtifact.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.config_entity import DataTransformationConfig
from src.config.artifacts_entity import DataIngestionArtifact, DataTransformationArtifact
from src import constants as training_pipeline
from utils.exceptions.exceptions import CustomerChurnPrediction
from utils.logging.logger import logging


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomerChurnPrediction(e, sys)

    def _get_feature_columns(self, df: pd.DataFrame):
        """Resolve categorical and numerical columns present in the dataframe."""
        target = training_pipeline.TARGET_COLUMN
        cat_candidates = [c for c in training_pipeline.CATEGORICAL_FEATURES if c in df.columns]
        num_candidates = [c for c in training_pipeline.NUMERICAL_FEATURES if c in df.columns]
        # If constants don't cover all, infer from dtypes (excluding target)
        feature_cols = [c for c in df.columns if c != target]
        for c in feature_cols:
            if c in cat_candidates or c in num_candidates:
                continue
            if pd.api.types.is_object_dtype(df[c]) or df[c].dtype.name == "category":
                cat_candidates.append(c)
            elif pd.api.types.is_numeric_dtype(df[c]):
                num_candidates.append(c)
        return cat_candidates, num_candidates

    def _prepare_data(self, df: pd.DataFrame):
        """Ensure numerical columns are numeric (e.g. TotalCharges)."""
        target = training_pipeline.TARGET_COLUMN
        for col in training_pipeline.NUMERICAL_FEATURES:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        return df

    def _build_preprocessor(self, categorical_features: list, numerical_features: list):
        """Build a single ColumnTransformer with OHE and StandardScaler."""
        try:
            transformers = []
            if numerical_features:
                transformers.append(
                    ("num", StandardScaler(), numerical_features)
                )
            if categorical_features:
                transformers.append(
                    (
                        "cat",
                        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                        categorical_features,
                    )
                )
            if not transformers:
                raise ValueError("No feature columns found for transformation.")
            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            logging.info(
                "Built preprocessor: numerical=%s, categorical=%s",
                numerical_features,
                categorical_features,
            )
            return preprocessor
        except Exception as e:
            raise CustomerChurnPrediction(e, sys)

    def _save_preprocessor(self, preprocessor: ColumnTransformer):
        """Save the fitted preprocessor as a pkl file."""
        try:
            path = self.data_transformation_config.transformed_object_file_path
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)
            joblib.dump(preprocessor, path)
            logging.info("Saved preprocessor to %s", path)
        except Exception as e:
            raise CustomerChurnPrediction(e, sys)

    def _save_transformed_arrays(self, X_train: np.ndarray, X_test: np.ndarray):
        """Save transformed feature matrices as npy files."""
        try:
            train_path = self.data_transformation_config.transformed_train_file_path
            test_path = self.data_transformation_config.transformed_test_file_path
            for path in (train_path, test_path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(train_path, X_train)
            np.save(test_path, X_test)
            logging.info("Saved transformed train and test arrays.")
        except Exception as e:
            raise CustomerChurnPrediction(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Load train/test from data_ingestion artifacts, build and fit preprocessor
        (OHE + StandardScaler), save preprocessor as pkl and transformed arrays as npy,
        then return DataTransformationArtifact.
        """
        try:
            train_path = self.data_ingestion_artifact.trained_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = self._prepare_data(train_df)
            test_df = self._prepare_data(test_df)

            target = training_pipeline.TARGET_COLUMN
            X_train = train_df.drop(columns=[target])
            X_test = test_df.drop(columns=[target])

            cat_cols, num_cols = self._get_feature_columns(X_train)
            preprocessor = self._build_preprocessor(cat_cols, num_cols)

            preprocessor.fit(X_train)

            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            self._save_preprocessor(preprocessor)
            self._save_transformed_arrays(X_train_transformed, X_test_transformed)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise CustomerChurnPrediction(e, sys)
