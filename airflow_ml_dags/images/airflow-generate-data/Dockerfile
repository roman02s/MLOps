FROM python:3.8.15-slim-buster
RUN python3 -m pip install --upgrade pip

COPY requirements.txt /

RUN pip3 install -r requirements.txt

COPY generate_data.py /generate_data.py

ENTRYPOINT ["python", "generate_data.py"]