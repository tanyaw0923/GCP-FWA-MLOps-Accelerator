FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app.py .

RUN mkdir -p /model

COPY model/model.bst /model/model.bst

ENV MODEL_PATH=/model/model.bst

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
