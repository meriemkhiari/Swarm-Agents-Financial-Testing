FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY shared /app/shared
COPY aggregator /app/aggregator

ENV PYTHONPATH=/app

EXPOSE 8005

HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('localhost', 8005), timeout=2.0)" || exit 1

CMD ["python", "-m", "aggregator.autogen_groupchat_aggregator"]