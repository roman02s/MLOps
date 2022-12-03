from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from utils import (
    GENERATED_DATA_PATH,
    PREDICTIONS_DATA_PATH,
    MODEL_PATH,
    MOUNT_SOURCE,
    default_args,
)

with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2022, 11, 30),
) as dag:
    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {GENERATED_DATA_PATH} --model-dir {MODEL_PATH} --output-dir {PREDICTIONS_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        auto_remove=True,
        mounts=[MOUNT_SOURCE],
    )

    predict
