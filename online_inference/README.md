
MLOps hw2 online_inference
==============================
Installation:

You can make image locally or load image from docker-hub.

Locally:
~~~
docker build -t server-classifier:v1 .
docker run -p 8000:8000 server-classifier:v1
~~~
From docker-hub:
~~~
docker pull naumenkokm/server-classifier:v1
docker run -p 8000:8000 naumenkokm/server-classifier:v1
~~~
To check, that server works use:
~~~
127.0.0.1:8000/health
~~~

To get prediction you could send a POST request with data-file or use client.py:
~~~
python client.py DATA_PATH PREDICTION_PATH
~~~
Example:
~~~
python client.py data/test_data.csv prediction.csv
~~~





Project Organization
------------
    ├── app
    │   ├── app.py
    │   ├── client.py
    │   └── test_main.py
    ├── data
    │   ├── heart_cleveland_upload.csv
    │   └── test_data.csv
    ├── Dockerfile
    ├── model
    │   └── model.pkl
    ├── README.md
    └── requirements.txt
--------