FROM python:3.10-buster

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY clairvoyance /clairvoyance
COPY setup.py /setup.py
RUN pip install .

CMD uvicorn clairvoyance.api.fast:app --host 0.0.0.0 --port $PORT
