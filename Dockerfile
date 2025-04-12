# Use a lightweight Python image as the base
FROM python:3.12-slim

# Change to the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI application code into the container
COPY . /app

# Expose port 5000 for the FastAPI server
EXPOSE 5000

# By default, run the API with Uvicorn on port 5000
CMD ["uvicorn", "api_fastapi:app", "--host", "0.0.0.0", "--port", "5000"]
