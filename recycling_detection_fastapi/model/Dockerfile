# FROM nvcr.io/nvidia/pytorch:24.05-py3     # If GPU available
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your specific model file
COPY loopvision_2025Jun25.pth .
COPY inference.py .

# Command to run your inference script when the container starts
# This will execute the `if __name__ == "__main__":` block in inference.py
CMD ["python", "inference.py"]

# # Serving as a REST API:
# # Install FastAPI and uvicorn
# RUN pip install --no-cache-dir fastapi uvicorn
# # ...
# CMD ["uvicorn", "-w", "1", "-b", "0.0.0.0:5000", "inference:app"] 
# # Assuming 'app' is your FastAPI instance in inference.py
# # EXPOSE 5000 (to make port 5000 accessible from outside the container)