FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:$PORT app:app"]