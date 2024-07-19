# Use the official Python image from the Docker Hub
FROM python:3.10-slim
RUN apt-get update && apt-get install -y gcc build-essential

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app runs on
EXPOSE 8000

# Run main.py with Python
CMD ["python", "main.py"]
