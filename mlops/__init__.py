"""MLOps library for data science portfolio apps.

Provides reusable components for:
- Data loading and validation
- Model training and evaluation
- Drift detection
- Feature engineering
- Model persistence
"""

from .data_loader import DataLoader
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .drift_detector import DriftDetector
from .feature_engineer import FeatureEngineer
from .model_persistor import ModelPersistor

__all__ = [
    "DataLoader",
    "ModelTrainer",
    "ModelEvaluator",
    "DriftDetector",
    "FeatureEngineer",
    "ModelPersistor",
]

__version__ = "1.0.0"
