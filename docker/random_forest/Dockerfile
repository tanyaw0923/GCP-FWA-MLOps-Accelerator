FROM python:3.11-slim

WORKDIR /app

COPY docker/random_forest/requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --upgrade \
       'jaraco.context>=6.1.0' 'wheel>=0.46.2'

COPY src ./src

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.models.run_random_forest_training"]
