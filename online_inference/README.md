Online inference модели проекта для решения задачи классификации
==============================

Проект для решения задачи классификации.

Для обучения модели использовался датасет
https://www.kaggle.com/datasets/muhammad4hmed/monkeypox-patients-dataset
### Сборка docker:
    
    docker build -t roman02/fastapi:v1 .

### Публикация образа в https://hub.docker.com/

    docker push roman02/fastapi:v1

### Загрузка из https://hub.docker.com/ и запуск сервиса в docker:

    docker pull roman02/fastapi:v1
    
    docker run --rm -it -p 8000:8000 --name fastapi_container roman02/fastapi:v1

### Запрос к сервису(из директории online_inference):

    python make_request.py

### Запуск тестов:

    python -m pytest tests/tests.py

### Оптимизация размера docker image:
* Заменил базовый образ python:3.6 на более легкий python:3.6.15-slim-buster: уменьшение размера образа на 700 МБ
* Уменьшил количество слоев в Dockerfile: уменьшение размера образа на 2МБ 
* Добавил --no-cache-dir: уменьшение размера образа на 3МБ
* Добавлял команды по мере их изменчивости: не меняющиеся команды помещал выше, команды, склонные к изменчивости, добавлял ближе к концу dockerfile
