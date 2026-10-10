FROM python:3.11-slim

WORKDIR /app

COPY docker/random_forest/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.models.run_random_forest_training"]
