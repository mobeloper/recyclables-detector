# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# IMPORTANT: Set this environment variable for MPS fallback
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Install missing system-level dependencies for OpenCV (libGL.so.1)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Add other common dependencies if issues arise later
    # libsm6 \
    # libxrender1 \
    # libxext6 \
    # libfontconfig1 \
    # libxft6 \
    # libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your specific model file
COPY loopvision_2025Jun25.pth .

# Copy your FastAPI application
COPY main.py .

# Expose the port your FastAPI app will listen on
EXPOSE 5050

# Command to run your FastAPI app with Gunicorn managing Uvicorn workers
# -w: number of worker processes (start with 1 or `(2 * CPU_CORES) + 1` for optimal CPU usage)
# -k uvicorn.workers.UvicornWorker: tells Gunicorn to use Uvicorn as its worker class
# main:app refers to the 'app' variable in main.py
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:5050", "main:app"]