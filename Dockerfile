# Base image
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 gcc

# Set work directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["gunicorn", "backend:app", "--bind", "0.0.0.0:5000"]
