FROM python:3.8.15-slim-buster
RUN python3 -m pip install --upgrade pip

COPY requirements.txt /

RUN pip3 install -r requirements.txt

COPY predict.py /predict.py

ENTRYPOINT ["python3", "predict.py"]
