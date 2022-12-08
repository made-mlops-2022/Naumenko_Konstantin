FROM airflow-ml-base:latest

COPY download.py /download.py
COPY requirements.txt /build/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /build/requirements.txt

ENTRYPOINT ["python", "download.py"]