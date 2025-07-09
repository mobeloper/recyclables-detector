# FROM nvcr.io/nvidia/pytorch:24.05-py3     # If GPU available

# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# IMPORTANT: Install missing system-level dependencies for OpenCV (libGL.so.1)
# and other common libraries that might be needed by vision tasks.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # You might also need these, though libgl1-mesa-glx is the primary one for libGL.so.1
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

# Copy your inference script
COPY inference.py .

# Command to run your inference script
CMD ["python", "inference.py"]
# # Serving as a REST API:
# # Install FastAPI and uvicorn
# RUN pip install --no-cache-dir fastapi uvicorn
# # ...
# CMD ["uvicorn", "-w", "1", "-b", "0.0.0.0:5000", "inference:app"] 
# # Assuming 'app' is your FastAPI instance in inference.py
# # EXPOSE 5000 (to make port 5000 accessible from outside the container)