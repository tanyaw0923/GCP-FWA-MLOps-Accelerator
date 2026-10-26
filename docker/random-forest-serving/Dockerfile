FROM python:3.11-slim

WORKDIR /app

COPY docker/random-forest-serving/requirements.txt ./requirements.txt

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY docker/random-forest-serving/app.py ./app.py

RUN mkdir -p /model

ENV MODEL_PATH=/model/model.joblib

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
