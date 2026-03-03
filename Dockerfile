FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

RUN mkdir -p /app/data

ENV DB_PATH=/app/data/sgldhelper.db

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import sqlite3; sqlite3.connect('/app/data/sgldhelper.db').execute('SELECT 1')" || exit 1

CMD ["python", "-m", "sgldhelper"]
