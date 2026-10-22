FROM python:3.11-slim

WORKDIR /app

COPY docker/performance-monitoring/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

CMD ["python", "-m", "src.monitoring.run_model_performance_monitoring"]
