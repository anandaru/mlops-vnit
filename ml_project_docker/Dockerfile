# Use Python 3.10 as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy all necessary files to the container
COPY ./api ./api
COPY ./src ./src
COPY ./data ./data
COPY ./requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the required ports
EXPOSE 8000 5001

# Start MLflow and FastAPI when the container runs
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5001 & uvicorn api.app:app --host 0.0.0.0 --port 8000"]

