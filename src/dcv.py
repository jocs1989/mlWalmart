from ydata_profiling import ProfileReport
import mlflow
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def data_validation_report(df, output_path="./../reports/data_profile.html"):
    logger.info("📊 Starting data validation report generation...")

    # Create folder if it does not exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📁 Verified output directory: {output_dir}")

    # Create profiling report
    profile = ProfileReport(df, title="Data Validation Report", explorative=True)
    profile.to_file(output_path)
    logger.info(f"✅ HTML report generated: {output_path}")

    # Log artifact to MLflow
    mlflow.log_artifact(output_path, artifact_path="data_validation")
    logger.info("📎 Report logged as artifact in MLflow")

    # Basic data quality metrics
    num_rows = len(df)
    num_columns = df.shape[1]
    missing_pct = df.isnull().mean().mean() * 100

    mlflow.log_metric("num_rows", num_rows)
    mlflow.log_metric("num_columns", num_columns)
    mlflow.log_metric("missing_values_pct", missing_pct)
    logger.info(f"📈 Logged metrics - rows: {num_rows}, columns: {num_columns}, NaNs: {missing_pct:.2f}%")
