FROM python:3.12-slim

# LightGBM требует libgomp для OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости отдельным слоем — пересобирается только при изменении requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Данные организаторов и кэш погоды едут в образ: проект работает офлайн
COPY . .

# Модели обучаются при сборке — образ приезжает готовым к прогнозу (~2 мин)
RUN python -m src.cli train

EXPOSE 8501

# По умолчанию поднимается панель оператора; CLI доступен через docker compose run
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
