FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY . .

EXPOSE 8050

ENV PYTHONPATH=/app
ENV PORT=8050

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 src.dashboard.app:server
