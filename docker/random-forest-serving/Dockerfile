FROM python:3.11-slim

WORKDIR /app

COPY docker/random-forest-serving/requirements.txt ./requirements.txt

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

# Upgrade setuptools (including vendored jaraco.context and wheel metadata).
RUN python -m pip install --no-cache-dir --upgrade \
    'setuptools>=82.0.0' 'wheel>=0.46.2'

COPY docker/random-forest-serving/app.py ./app.py

RUN mkdir -p /model

ENV MODEL_PATH=/model/model.joblib

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
