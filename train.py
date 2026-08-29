# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import uuid
from pathlib import Path

# The system temporary directory is not writable in this environment. Configure
# it before importing MLflow, which creates temporary model artifacts.
PROJECT_DIR = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_DIR / ".tmp"
TEMP_DIR.mkdir(exist_ok=True)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn
from mlflow.utils import file_utils as mlflow_file_utils


def create_writable_temp_dir():
    """Create a MLflow scratch directory without Windows' restricted 0o700 ACL."""
    while True:
        path = TEMP_DIR / f"mlflow-{uuid.uuid4().hex}"
        try:
            path.mkdir(mode=0o777)
            return str(path)
        except FileExistsError:
            continue


# Python 3.14 creates tempfile directories with a restrictive Windows ACL in
# this environment. MLflow normally uses tempfile.mkdtemp(), so replace only
# its factory with a writable, project-local equivalent.
mlflow_file_utils.create_tmp_dir = create_writable_temp_dir


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Keep experiment data writable and independent of the current shell.
    # A user-supplied MLFLOW_TRACKING_URI still takes precedence.
    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"
        )
    )

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = PROJECT_DIR / "wine-quality.csv"
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    with mlflow.start_run():
        lr = RandomForestRegressor(n_estimators=200, random_state=42)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Random Forest model (n_estimators=200, random_state=42):")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, name="model")
