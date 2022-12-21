FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /server_classifier
COPY requirements.txt ./requirements.txt
COPY app/app.py ./app/app.py
COPY app/test_main.py ./app/test_main.py
COPY data/test_data.csv ./data/test_data.csv

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python ./app/test_main.py
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]