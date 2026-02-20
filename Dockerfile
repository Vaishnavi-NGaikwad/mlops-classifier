FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/
COPY static/ static/

RUN apt-get update && apt-get install -y wget && \
    mkdir -p model && \
    wget https://github.com/Vaishnavi-NGaikwad/mlops-classifier/releases/download/v1.0/simple_cnn_baseline_exp1_20260217_053749_best.pt \
    -O model/simple_cnn_baseline_exp1_20260217_053749_best.pt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]