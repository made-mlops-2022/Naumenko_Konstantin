FROM python:3.6
RUN mkdir /build/
COPY split.py /split.py
COPY train.py /train.py
COPY validate.py /validate.py
COPY requirements.txt /build/requirements.txt

RUN pip install -r /build/requirements.txt
