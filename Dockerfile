FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY data ./data

EXPOSE $PORT

CMD ["sh", "-c", "python src/main.py  --server.port $PORT"]