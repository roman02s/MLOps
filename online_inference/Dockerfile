FROM python:3.6.15-slim-buster

ARG WORK_DIR=/app
ENV PORT=8000 \
    HOST=0.0.0.0

WORKDIR ${WORK_DIR}
EXPOSE 8000

COPY . ${WORK_DIR}/
RUN python -m pip install --upgrade pip &&\
    pip install --no-cache-dir -r ${WORK_DIR}/requirements.txt

CMD set -xe; python app.py
