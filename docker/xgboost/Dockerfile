FROM python:3.11-slim

WORKDIR /app

COPY docker/xgboost/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.models.run_xgboost_training"]
