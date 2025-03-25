# Use an official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script and model file into the container
COPY lstm.py .
COPY model_lstm.pth .

# Default command to run FastAPI app with uvicorn
CMD ["uvicorn", "lstm:app", "--host", "0.0.0.0", "--port", "8000"]
