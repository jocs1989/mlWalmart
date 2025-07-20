import pandas as pd
import logging
from datetime import datetime
import kagglehub
import shutil
import os
import time
# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # prints to console
)

logger = logging.getLogger(__name__)


def download_and_move_dataset(dataset_name: str, custom_path: str):
    """
    Función para descargar un dataset usando kagglehub y moverlo a una ruta personalizada.
    
    Parámetros:
    - dataset_name (str): Nombre del dataset en Kaggle (ejemplo: 'arnabbiswas1/microsoft-azure-predictive-maintenance').
    - custom_path (str): Ruta donde se desea mover el dataset descargado.
    
    Retorna:
    - path (str): Ruta del dataset descargado.
    """
    
    try:
        # Descargar el dataset en la ruta predeterminada
        logger.info(f"Descargando dataset {dataset_name}...")
        path = kagglehub.dataset_download(dataset_name)

        # Verifica si la carpeta de destino existe, si no, la crea
        if not os.path.exists(custom_path):
            logger.info(f"Creando carpeta {custom_path}...")
            os.makedirs(custom_path)

        # Mueve los archivos descargados a la ruta especificada
        logger.info(f"Moviendo los archivos a {custom_path}...")
        shutil.move(path, custom_path)
        time.sleep(2)

        logger.info(f"Dataset files have been moved to: {custom_path}")
        return custom_path
    except Exception as e:
        logger.error(f"Error al descargar o mover el dataset: {e}")
        return None




def load_data(data_path='data/raw/3'):
    # Ejemplo de uso de la función
    dataset_name = "arnabbiswas1/microsoft-azure-predictive-maintenance"
    custom_path = "../data/raw"

    # Llamar la función para descargar y mover el dataset
    download_and_move_dataset(dataset_name, custom_path)

    logger.info(f"📥 Loading data from: {data_path}")

    telemetry = pd.read_csv(f'{data_path}/PdM_telemetry.csv')
    logger.info(f"✔️ telemetry.csv loaded: {telemetry.shape}")

    errors = pd.read_csv(f'{data_path}/PdM_errors.csv')
    logger.info(f"✔️ errors.csv loaded: {errors.shape}")

    maint = pd.read_csv(f'{data_path}/PdM_maint.csv')
    logger.info(f"✔️ maint.csv loaded: {maint.shape}")

    failures = pd.read_csv(f'{data_path}/PdM_failures.csv')
    logger.info(f"✔️ failures.csv loaded: {failures.shape}")

    machines = pd.read_csv(f'{data_path}/PdM_machines.csv')
    logger.info(f"✔️ machines.csv loaded: {machines.shape}")

    return telemetry, errors, maint, failures, machines


def preprocess_data(telemetry, errors, maint, failures, machines):
    logger.info("🔧 Starting data preprocessing...")

    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
    failures['datetime'] = pd.to_datetime(failures['datetime'])
    logger.info("📅 Converted date columns to datetime")

    df = telemetry.copy().sort_values(['machineID', 'datetime'])
    logger.info(f"📊 Data sorted: {df.shape}")

    df['failure_in_next_24h'] = 0
    failure_set = set(zip(failures['machineID'], failures['datetime']))
    logger.info(f"📌 Failure set prepared: {len(failure_set)} entries")

    # Create labels
    total_rows = len(df)
    for idx, row in df.iterrows():
        machine = row['machineID']
        time = row['datetime']
        window = pd.date_range(time, periods=24, freq='h')

        if any((machine, t) in failure_set for t in window):
            df.at[idx, 'failure_in_next_24h'] = 1

        if idx % 10000 == 0:  # Log every 10k rows
            logger.info(f"Labeling in progress: {idx}/{total_rows}")

    logger.info("✅ Failure labels generated")

    # Enrich with machine metadata
    df = df.merge(machines, on='machineID', how='left')
    logger.info(f"🔗 Merged with machines.csv completed: {df.shape}")

    return df
