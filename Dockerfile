# Use an official Python runtime
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy app source code
COPY app/ /app/

# Copy model directory into the container
COPY model/ /app/model/

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose port 8000
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



