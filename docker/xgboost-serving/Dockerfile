FROM python:3.10-slim

# Apply available Debian security fixes, including apt/libapt-pkg7.0.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/xgboost-serving/requirements.txt ./requirements.txt

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

# Upgrade setuptools (including vendored jaraco.context and wheel metadata).
RUN python -m pip install --no-cache-dir --upgrade \
    'setuptools>=82.0.0' 'wheel>=0.46.2'

COPY docker/xgboost-serving/app.py ./app.py

RUN mkdir -p /model

ENV MODEL_PATH=/model/model.bst

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
