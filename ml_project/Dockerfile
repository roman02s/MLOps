FROM python:3.6

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt


COPY configs /configs
RUN mkdir -p /models

WORKDIR .

CMD ["train_pipeline", "configs/train_config.yaml"]
