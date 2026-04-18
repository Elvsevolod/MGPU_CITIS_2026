FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt первым (для кэширования слоя при его изменении)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем все остальные файлы приложения
COPY dashboard_v2.py .
COPY hakaton.csv .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "dashboard_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]