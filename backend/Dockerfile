FROM python:3.11

WORKDIR /app

RUN pip3 install poetry
RUN poetry config virtualenvs.create false

COPY poetry.lock pyproject.toml /app/

RUN poetry install

COPY . /app/