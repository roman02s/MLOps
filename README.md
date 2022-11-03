# Машинное обучение в продакшене

Сим Роман Дмитриевич

Группа ML-21

Преподаватели: Михаил Марюфич

# Homework 1

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
~~~
python ml_example/train_pipeline.py configs/train_config.yaml
~~~

Test:
~~~
pytest tests/
~~~

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml_example                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    │   │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


DOCKER:
~~~
python setup.py sdist
docker build -t mikhailmar/train_made:v1 
docker run -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} mikhailmar/train_made:v1
~~~


Проект для решения задачи классификации.

Для обучения модели использовался датасет 

[//]: # (https://www.kaggle.com/datasets/whenamancodes/students-performance-in-exams)
https://www.kaggle.com/datasets/muhammad4hmed/monkeypox-patients-dataset

**Критерии (указаны максимальные баллы, по каждому критерию ревьюер может поставить баллы частично):**

0) В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание того, что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код (1 балл)
1) В пулл-реквесте проведена самооценка, распишите по каждому пункту выполнен ли критерий или нет и на сколько баллов(частично или полностью) (1 балл)

2) Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1 балл)
   Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы)

   Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это + 1 балл)

3) Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (3 балла)
4) Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3 балла)

5) Проект имеет модульную структуру (2 балла)
6) Использованы логгеры (2 балла)

7) Написаны тесты на отдельные модули и на прогон обучения и predict (3 балла)

8) Для тестов генерируются синтетические данные, приближенные к реальным (2 балла)
   - можно посмотреть на библиотеки https://faker.readthedocs.io/, https://feature-forge.readthedocs.io/en/latest/
   - можно просто руками посоздавать данных, собственноручно написанными функциями.

9) Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)
10) Используются датаклассы для сущностей из конфига, а не голые dict (2 балла)

11) Напишите кастомный трансформер и протестируйте его (3 балла)
   https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

12) В проекте зафиксированы все зависимости (1 балл)
13) Настроен CI для прогона тестов, линтера на основе github actions (3 балла).
Пример с пары: https://github.com/demo-ml-cicd/ml-python-package

PS: Можно использовать cookiecutter-data-science  https://drivendata.github.io/cookiecutter-data-science/ , но поудаляйте папки, в которые вы не вносили изменения, чтобы не затруднять ревью

Дополнительные баллы=)
- Используйте hydra для конфигурирования (https://hydra.cc/docs/intro/) - 3 балла

Mlflow
- разверните локально mlflow или на какой-нибудь виртуалке (1 балл)
- залогируйте метрики (1 балл)
- воспользуйтесь Model Registry для регистрации модели(1 балл)
  Приложите скриншот с вашим mlflow run
  DVC
- выделите в своем проекте несколько entrypoints в виде консольных утилит (1 балл).
  Пример: https://github.com/made-ml-in-prod-2021/ml_project_example/blob/main/setup.py#L16
  Но если у вас нет пакета, то можно и просто несколько скриптов

- добавьте датасет под контроль версий (1 балл)
- сделайте dvc пайплайн(связывающий запуск нескольких entrypoints) для изготовления модели(1 балл)

Для большего удовольствия в выполнении этих частей рекомендуется попробовать подключить удаленное S3 хранилище(например в Yandex Cloud, VK Cloud Solutions или Selectel)

**Процедура сдачи**:

После выполнения ДЗ создаем пулл реквест, в ревьюеры добавляем  Mikhail-M, ждем комментариев (на которые нужно ответить) и/или оценки.

**Сроки выполнения**:

Мягкий дедлайн: 9 мая 23:59

Жесткий дедлайн:  16 мая 23:59

После мягкого дедлайна все полученные баллы умножаются на 0.6