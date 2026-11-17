FROM python:3.13
WORKDIR /usr/local/app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY src ./src
EXPOSE 8000

# Setup an app user so the container doesn't run as the root user
RUN useradd our_group_app
USER our_group_app

CMD ["uvicorn", "src.deployment.fastapi:app", "--host", "0.0.0.0", "--port", "8000"]