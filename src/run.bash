#!/bin/bash

# Función para manejar errores y salidas tempranas
function error_exit {
    echo -e "\033[0;31m❌ ERROR: $1\033[0m"
    exit 1
}

function success_message {
    echo -e "\033[0;32m✅ $1\033[0m"
}

function info_message {
    echo -e "\033[0;34m⚙️ $1\033[0m"
}

# Verificar si el script está siendo ejecutado como root
if [[ $EUID -ne 0 ]]; then
   info_message "⚠️ Este script no está siendo ejecutado como root. Algunos comandos pueden requerir permisos elevados."
   read -p "¿Deseas continuar como usuario normal? (y/n) " CONTINUAR
   if [[ ! "$CONTINUAR" =~ ^[Yy]$ ]]; then
       error_exit "Por favor, ejecuta el script como root para continuar."
   fi
fi

echo -e "\033[0;34m🚀 Starting project setup validation...\033[0m"

# Verificar sistema operativo
OS=$(uname)
if [[ "$OS" != "Linux" ]]; then
  error_exit "This script only supports Linux. You are using: $OS"
fi
success_message "Operating system is Linux."

# Verificar Anaconda/conda
if ! command -v conda &> /dev/null; then
  error_exit "Anaconda/conda not found. Please install Anaconda before proceeding."
fi
success_message "Anaconda (conda) is installed."

# Verificar Docker
if ! command -v docker &> /dev/null; then
  error_exit "Docker not found. Please install Docker before proceeding."
fi
success_message "Docker is installed."

# Verificar conexión a internet (probando con un servidor público)
if ! curl -s google.com > /dev/null; then
  error_exit "No internet connection. Please ensure you are connected to the internet."
fi
success_message "Internet connection verified."

echo -e "\033[0;32m🎉 All requirements met! You can now start the project.\033[0m"

# Verificar si el archivo conda.yaml existe
if [ ! -f "conda.yaml" ]; then
  error_exit "File 'conda.yaml' not found. Please ensure the 'conda.yaml' file is in the current directory."
fi

# Crear entorno Conda usando el archivo conda.yaml
info_message "Creating conda environment from 'conda.yaml'..."
conda env create -f conda.yaml || error_exit "Failed to create conda environment from 'conda.yaml'"

# Verificar si el entorno de Conda se creó correctamente
ENV_NAME=$(awk '/name:/ {print $2}' conda.yaml)
if ! conda info --envs | grep -q "$ENV_NAME"; then
    error_exit "The conda environment '$ENV_NAME' could not be created. Please check the 'conda.yaml' file."
fi
success_message "Conda environment '$ENV_NAME' created successfully."

# Activar entorno conda
info_message "Activating conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh" || error_exit "Failed to source conda"
conda activate "$ENV_NAME" || error_exit "Failed to activate conda environment '$ENV_NAME'"

success_message "Environment '$ENV_NAME' is activated. Ready to go!"

# Verificar si las dependencias necesarias están instaladas
info_message "Checking if 'mlflow' is installed in the environment..."
if ! conda list | grep -q "mlflow"; then
  info_message "Installing mlflow..."
  conda install -n "$ENV_NAME" -c conda-forge mlflow || error_exit "Failed to install mlflow"
  success_message "mlflow installed successfully."
else
  success_message "mlflow is already installed."
fi

# Verificar espacio en disco antes de ejecutar Docker
info_message "Checking disk space..."
AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
if [ "$AVAILABLE_SPACE" -lt 1000000 ]; then  # Si el espacio disponible es menor a 1GB
  error_exit "Insufficient disk space. Please free up space and try again."
fi
success_message "Disk space is sufficient."

echo -e "\033[0;34m⏳ Starting training with 'ml-walmart'...\033[0m"

# Ejecutar el script de entrenamiento dentro del entorno activado de Conda
python make.py || error_exit "Training script failed to execute"

echo -e "\033[0;32m🎉 Training completed successfully!\033[0m"

echo -e "\033[0;34m⏳ Serving model 'ml-walmart'...\033[0m"

# Verificar si Docker está funcionando correctamente antes de ejecutar el contenedor
if ! docker info &> /dev/null; then
  error_exit "Docker is not running or not configured correctly. Please start Docker and try again."
fi

docker run -p 5001:8080 ml-faild-prediction || error_exit "Failed to start Docker container."

success_message "Model 'ml-walmart' is now serving at port 5001!"
