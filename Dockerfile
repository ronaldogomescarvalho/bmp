FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libxcb1 libx11-6 libgl1 libglib2.0-0 \
    libgles2 libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]