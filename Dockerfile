FROM python:3.13-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ serving/

EXPOSE 5001

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "5001"]