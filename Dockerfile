# Образ с проверенным комплектом: код, данные организаторов, кэш погоды, сохранённые
# модели и канонические прогнозы. Обучения при сборке нет: образ должен показывать
# ровно те числа, которые интегратор проверил и зафиксировал в manifest.json,
# а переобучение в другой среде дало бы иной артефакт без повторной оценки.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# LightGBM требует libgomp (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости отдельным слоем: пересобирается только при изменении requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Всё остальное, включая models_artifacts/ и forecasts/ (исключения — в .dockerignore)
COPY . .

# Проверка комплекта при сборке, без сети: сохранённые модели читаются установленным
# sklearn без предупреждения о несовместимой версии, каноническая подача цела.
RUN python -c "import warnings; from sklearn.exceptions import InconsistentVersionWarning; \
      warnings.simplefilter('error', InconsistentVersionWarning); \
      from src.models.predict import load_model; load_model(1); load_model(2)" \
    && python -m scripts.verify_submission

EXPOSE 8501

# Готовность панели: /_stcore/health отвечает 200, когда Streamlit поднялся
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else 1)"

# По умолчанию — панель оператора; offline-репетиция: профиль check в docker-compose.yml
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
