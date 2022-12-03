from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from utils import (
    GENERATED_DATA_PATH,
    MOUNT_SOURCE,
    default_args,
)


with DAG(
    "generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2022, 11, 30),
) as dag:
    generate_data = DockerOperator(
        image="airflow-generate-data",
        command=f"--output-dir {GENERATED_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-generate-data",
        do_xcom_push=False,
        auto_remove=True,
        mounts=[MOUNT_SOURCE],
    )

    generate_data
