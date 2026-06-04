"""Model persistor module for saving and loading models."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class ModelPersistor:
    """Save and load ML models with metadata."""

    def __init__(self, model_dir: str = "models"):
        """Initialize model persistor.

        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "pickle",
    ) -> Path:
        """Save model to disk.

        Args:
            model: Model object (sklearn compatible)
            name: Model name
            metadata: Optional metadata to save with model
            format: Save format ('pickle' or 'joblib')

        Returns:
            Path to saved model
        """
        filename = f"{name}.pkl"
        filepath = self.model_dir / filename

        if format == "joblib":
            try:
                import joblib
                joblib.dump(model, filepath)
            except ImportError:
                import pickle
                with open(filepath, "wb") as f:
                    pickle.dump(model, f)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(model, f)

        # Save metadata
        if metadata is not None:
            meta_path = self.model_dir / f"{name}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        return filepath

    def load(
        self,
        name: str,
        format: str = "pickle",
    ) -> Any:
        """Load model from disk.

        Args:
            name: Model name
            format: Save format ('pickle' or 'joblib')

        Returns:
            Loaded model
        """
        filepath = self.model_dir / f"{name}.pkl"

        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        if format == "joblib":
            try:
                import joblib
                return joblib.load(filepath)
            except ImportError:
                with open(filepath, "rb") as f:
                    return pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                return pickle.load(f)

    def save_metadata(
        self,
        name: str,
        metadata: Dict[str, Any],
    ) -> Path:
        """Save metadata separately.

        Args:
            name: Model name
            metadata: Metadata dictionary

        Returns:
            Path to metadata file
        """
        meta_path = self.model_dir / f"{name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return meta_path

    def load_metadata(
        self,
        name: str,
    ) -> Dict[str, Any]:
        """Load metadata for a model.

        Args:
            name: Model name

        Returns:
            Metadata dictionary
        """
        meta_path = self.model_dir / f"{name}_meta.json"

        if not meta_path.exists():
            return {}

        with open(meta_path, "r") as f:
            return json.load(f)

    def list_models(self) -> list:
        """List all saved models.

        Returns:
            List of model names
        """
        models = []
        for filepath in self.model_dir.glob("*.pkl"):
            models.append(filepath.stem.replace("_meta", ""))
        return list(set(models))

    def delete(self, name: str) -> bool:
        """Delete a model and its metadata.

        Args:
            name: Model name

        Returns:
            True if deleted, False if not found
        """
        filepath = self.model_dir / f"{name}.pkl"
        meta_path = self.model_dir / f"{name}_meta.json"

        deleted = False
        if filepath.exists():
            filepath.unlink()
            deleted = True

        if meta_path.exists():
            meta_path.unlink()

        return deleted
