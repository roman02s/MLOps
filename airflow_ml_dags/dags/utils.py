import os
from datetime import timedelta

from airflow.models import Variable
from docker.types import Mount


LOCAL_DATA_DIR = Variable.get("LOCAL_DATA_DIR")
GENERATED_DATA_PATH = "/data/raw/{{ ds }}"
PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
PREDICTIONS_DATA_PATH = "/data/predictions/{{ ds }}"
MODEL_PATH = "/data/models/lr/{{ ds }}"
VOLUME_PATH = "/opt/airflow"
MOUNT_SOURCE = Mount(
    source="/Users/romansim/Проекты/Технопарк/ML-разработчик/2 семестр/Машинное обучение в продакшен/homeworks/airflow_ml_dags/data/",
    target="/data",
    type='bind'
)


def custom_failure_function(context):
    dag_run = context.get('dag_run')
    task_instances = dag_run.get_task_instances()
    print("These task instances failed:", task_instances)


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "on_failure_callback": custom_failure_function
}


def wait_file(input_file: str) -> bool:
    return os.path.exists(input_file)
