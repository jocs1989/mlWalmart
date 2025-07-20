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
COPY . /app/.

# Comando por defecto vacío, Kubeflow decidirá qué ejecutar
# Comando por defecto: ejecutar el script Python que estará en 'src/'
CMD ["bash", "-c", "cd src && bash run.bash"]
