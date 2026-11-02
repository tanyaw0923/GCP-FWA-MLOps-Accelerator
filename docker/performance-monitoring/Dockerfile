FROM python:3.11-slim

WORKDIR /app

COPY docker/performance-monitoring/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --upgrade \
       'setuptools>=82.0.0' 'wheel>=0.46.2'

COPY src ./src

CMD ["python", "-m", "src.monitoring.run_model_performance_monitoring"]
