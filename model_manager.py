"""
Model Manager - Handle model versioning, tracking, and management
"""
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path


class ModelManager:
    """Manage model versions, metadata, and training history."""

    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Load model metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "current": None}

    def _save_metadata(self):
        """Save model metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_model(self, model_path, accuracy, loss, training_time, epochs, batch_size):
        """Register a trained model with metadata."""
        model_hash = self._compute_hash(model_path)
        version = len(self.metadata["models"]) + 1
        model_id = f"v{version}"

        self.metadata["models"][model_id] = {
            "version": version,
            "path": str(model_path),
            "hash": model_hash,
            "accuracy": accuracy,
            "loss": loss,
            "training_time_seconds": training_time,
            "epochs": epochs,
            "batch_size": batch_size,
            "created_at": datetime.now().isoformat(),
            "size_mb": os.path.getsize(model_path) / (1024 * 1024)
        }

        # Set as current model
        self.metadata["current"] = model_id
        self._save_metadata()

        print(f"Model registered: {model_id}")
        return model_id

    def get_current_model(self):
        """Get current model path."""
        current = self.metadata.get("current")
        if current and current in self.metadata["models"]:
            return self.metadata["models"][current]["path"]
        return None

    def get_model_info(self, model_id=None):
        """Get model information."""
        if model_id is None:
            model_id = self.metadata.get("current")

        if model_id in self.metadata["models"]:
            return self.metadata["models"][model_id]
        return None

    def list_models(self):
        """List all registered models."""
        return self.metadata["models"]

    def _compute_hash(self, file_path):
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:8]
