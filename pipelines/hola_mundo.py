from kfp import dsl
from kfp import compiler
from kfp.dsl import ContainerSpec

@dsl.component(
    base_image='jocz/ml-conda:latest'  # No usar 'docker pull' aquí
)
def componente_con_conda_yaml():
    print("Hello")

@dsl.pipeline(name='pipeline-con-conda-yaml')
def pipeline_conda_yaml():
    componente_con_conda_yaml()

if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_conda_yaml, 'pipeline_conda_yaml.yaml')