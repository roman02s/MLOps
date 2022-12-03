from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor

from utils import (
    GENERATED_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    VOLUME_PATH,
    MOUNT_SOURCE,
    default_args,
    wait_file,
)


with DAG(
    "train",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2022, 11, 30),
) as dag:
    wait_data = PythonSensor(
        task_id="airflow-wait-data",
        python_callable=wait_file,
        op_args=[VOLUME_PATH + GENERATED_DATA_PATH + "/data.csv"],
    )

    wait_target = PythonSensor(
        task_id="airflow-wait-target",
        python_callable=wait_file,
        op_args=[VOLUME_PATH + GENERATED_DATA_PATH + "/target.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {GENERATED_DATA_PATH} --output-dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        auto_remove=True,
        mounts=[MOUNT_SOURCE]
    )

    split_data = DockerOperator(
        image="airflow-split",
        command=f"--input-dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        auto_remove=True,
        mounts=[MOUNT_SOURCE]
    )

    train_model = DockerOperator(
        image="airflow-train",
        command=f"--input-dir {PROCESSED_DATA_PATH} --output-dir {MODEL_PATH}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        auto_remove=True,
        mounts=[MOUNT_SOURCE]
    )

    validate_model = DockerOperator(
        image="airflow-validate",
        command=f"--input-dir {PROCESSED_DATA_PATH} --model-dir {MODEL_PATH} --output-dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        auto_remove=True,
        mounts=[MOUNT_SOURCE]
    )

    [wait_data, wait_target] >> preprocess >> split_data >> train_model >> validate_model
