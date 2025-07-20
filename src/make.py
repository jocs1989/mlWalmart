import logging
import os
import mlflow
import mlflow.sklearn
from mlflow.models import build_docker
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from data import load_data, preprocess_data
from dcv import data_validation_report
from features import feature_engineering
import pandas as pd
import matplotlib.pyplot as plt
import subprocess


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = "FailurePredictorModel"

def log_training_metrics_plot(acc_train, f1_train, output_dir="mlruns_artifacts"):
    """
    Generates a bar plot for training accuracy and F1 score,
    saves it as PNG, and logs it as an MLflow artifact.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "train_metrics.png")

    # Create plot
    plt.figure(figsize=(6, 4))
    plt.bar(["Train Accuracy", "Train F1 Score"], [acc_train, f1_train],
            color=["skyblue", "salmon"])
    plt.title("Training Metrics")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.tight_layout()

    # Save and log
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Training metrics plot saved to {plot_path}")

    mlflow.log_artifact(plot_path, artifact_path="plots")
    logger.info("Training metrics plot logged as MLflow artifact.")

def train_and_log_model(X_train, y_train, df):
    logger.info("Generating data validation report...")
    data_validation_report(df)

    n_estimators = 100
    max_depth = None
    logger.info(f"Training RandomForest model with n_estimators={n_estimators}, max_depth={max_depth}...")

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    clf.fit(X_train, y_train)

    preds_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, preds_train)
    f1_train = f1_score(y_train, preds_train)
    logger.info(f"Train metrics: Accuracy={acc_train:.4f}, F1 Score={f1_train:.4f}")

    preds_df = pd.DataFrame(preds_train, columns=["prediction"])
    signature = infer_signature(X_train, preds_df)
    input_example = X_train.head(1)
    # 💾 Opcional: guarda el input_example como JSON en disco
    input_json_path = "input_example.json"
    input_example.to_json(input_json_path, orient="records", lines=False)
    logger.info(f"Input example saved to '{input_json_path}' for API usage.")
    log_training_metrics_plot(acc_train, f1_train, output_dir="mlruns_artifacts")
    logger.info("Logging parameters, metrics, and model to MLflow...")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("train_accuracy", acc_train)
    mlflow.log_metric("train_f1_score", f1_train)
    conda_env_path = os.path.join(os.path.dirname(__file__), "conda.yaml")
    mlflow.sklearn.log_model(
        clf,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        signature=signature,
        input_example=input_example,
        conda_env=conda_env_path
    )
    logger.info("Model trained and registered successfully.")
    return clf ,input_example

def test_logged_model(run_id, X_test, y_test):
    logger.info(f"Loading model from MLflow run {run_id} for testing...")
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    logger.info("Making predictions on test set...")
    preds = loaded_model.predict(X_test)

    acc_test = accuracy_score(y_test, preds)
    f1_test = f1_score(y_test, preds)
    logger.info(f"Test metrics: Accuracy={acc_test:.4f}, F1 Score={f1_test:.4f}")

    mlflow.log_metric("test_accuracy", acc_test)
    mlflow.log_metric("test_f1_score", f1_test)
    classification_report(y_test, preds)
    logger.info("Test Classification Report:\n")

def deploy_model_and_predict(input_example):
    logger.info(f"Loading registered model '{MODEL_NAME}' for deployment simulation...")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

    # Example dummy input (simulate prediction)
    
    example_input = input_example
    if example_input is None:
        logger.warning("No input example found in model metadata, using dummy data.")
        # You could replace this with an actual example input with correct features
        example_input = pd.DataFrame([{
            'volt': 120,
            'rotate': 30,
            'pressure': 5,
            'vibration': 0.1,
            'volt_rolling_mean_3': 119,
            'volt_rolling_std_3': 0.5,
            'rotate_rolling_mean_3': 29,
            'rotate_rolling_std_3': 0.1,
            'pressure_rolling_mean_3': 5,
            'pressure_rolling_std_3': 0.2,
            'vibration_rolling_mean_3': 0.1,
            'vibration_rolling_std_3': 0.01,
            'model': 0,
            'hour': 12
        }])
    logger.info(f"Running model prediction on example input:\n{example_input}")
    prediction = model.predict(example_input)
    logger.info(f"Model prediction result: {prediction}")

def train_with_mlflow():
    logger.info("Loading data...")
    telemetry, errors, maint, failures, machines = load_data(data_path='../data/raw/3')

    logger.info("Preprocessing data...")
    df = preprocess_data(telemetry, errors, maint, failures, machines)

    logger.info("Applying feature engineering...")
    X, y = feature_engineering(df)

    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="RandomForestFailurePrediction") as run:
        clf,input_example = train_and_log_model(X_train, y_train, df)

        test_logged_model(run_id=run.info.run_id, X_test=X_test, y_test=y_test)

        deploy_model_and_predict(input_example=input_example)

        logger.info(f"MLflow run ID: {run.info.run_id}")
    return run.info.run_id




if __name__ == "__main__":
    run_id=train_with_mlflow()
    

    # Comando que quieres ejecutar
    command = [
        "mlflow", "models", "build-docker", 
        "-m", f"runs:/{run_id}/model", 
        "-n", "ml-faild-prediction"
    ]

    # Ejecutar el comando como subproceso
    try:
        logger.info("Build Docker")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info("Salida estándar:", result.stdout)
        logger.info("Error estándar:", result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al ejecutar el comando: {e}")

    
