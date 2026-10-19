"""Central configuration: paths, column schema and feature groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MODEL_FILE = MODELS_DIR / "heart_model.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Column order of the raw UCI file (no header row in the source).
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Continuous measurements -> imputed with median and standard-scaled.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Already 0/1 -> passed through after imputation.
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Nominal codes -> one-hot encoded.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

INPUT_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_EXPERIMENT = "heart-disease-uci"
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
