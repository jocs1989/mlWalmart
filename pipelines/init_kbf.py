import kfp

client = kfp.Client(host='http://localhost:45011')

run = client.create_run_from_pipeline_package(
    pipeline_file='/home/jocs/Documentos/ml/ml-walmart/pipelines/pipeline_conda_yaml.yaml',
    arguments={},  # No hay argumentos en este pipeline simple
    run_name='ejecucion-hola-mundo'
)

print(f"Run ID: {run.run_id}")
