# Usa la imagen oficial de miniconda
FROM continuumio/miniconda3

# Copia el archivo de entorno Conda
COPY conda.yaml /tmp/mlflow-env.yml

# Crea el entorno Conda desde el YAML
RUN conda env create -f /tmp/mlflow-env.yml

# Activa el entorno por defecto para cada sesión
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Directorio de trabajo
WORKDIR /app

# Copia el código que vas a ejecutar (opcional)
# COPY train.py /app/train.py

# Comando por defecto vacío, Kubeflow decidirá qué ejecutar

ENTRYPOINT ["/bin/bash"]
