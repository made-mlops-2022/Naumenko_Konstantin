import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(10),
        tags=['hw3']
) as dag:
    try:
        model_name = Variable.get("model_name")
    except:
        model_name = '2022-12-08'
    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/predicted/{{ ds }} --model-dir /data/models/" + model_name,
        task_id="predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/tmp", target="/data", type='bind')]
    )