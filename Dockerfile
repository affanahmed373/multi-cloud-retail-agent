FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]