from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

RANDOM_STATE = 42
TEST_SIZE = 0.2

SENSITIVE_COL = "sex"
TARGET_COL = "income"

# Reduced hyperparameter search space for notebooks
RHO_GRID = [0.01, 0.1, 1.0]
LAMBDA_GRID = [0.1, 0.5, 0.9]
THETA_GRID = [0.1, 0.5, 0.9]

# Expanded grids for closer paper-style tuning
RHO_GRID_EXT = [0.001, 0.01, 0.1, 1.0, 10.0]
LAMBDA_GRID_EXT = [0.1, 0.3, 0.5, 0.7, 0.9]
THETA_GRID_EXT = [0.1, 0.3, 0.5, 0.7, 0.9]
