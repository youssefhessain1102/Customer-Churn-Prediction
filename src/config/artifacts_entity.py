from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision: float
    recall: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    trained_metric_artifact: ClassificationMetricArtifact
    tested_metric_artifact: ClassificationMetricArtifact