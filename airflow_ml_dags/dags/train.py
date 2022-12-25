import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "train",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(14),
        tags=['hw3']
) as dag:
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/tmp", target="/data", type='bind')]
    )

    split = DockerOperator(
        image="airflow-ml-base",
        command="python split.py /data/processed/{{ ds }}",
        task_id="split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/tmp", target="/data", type='bind')]
    )

    train = DockerOperator(
        image="airflow-ml-base",
        command="python train.py /data/processed/{{ ds }} /data/models/{{ ds }}",
        task_id="train",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/tmp", target="/data", type='bind')]
    )

    validate = DockerOperator(
        image="airflow-ml-base",
        command="python validate.py /data/processed/{{ ds }} /data/models/{{ ds }}",
        task_id="validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/tmp", target="/data", type='bind')]
    )

    preprocess >> split >> train >> validate
