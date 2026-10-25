FROM python:3.10-slim

WORKDIR /app

COPY docker/xgboost-serving/requirements.txt ./requirements.txt

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY docker/xgboost-serving/app.py ./app.py

RUN mkdir -p /model

ENV MODEL_PATH=/model/model.bst

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
