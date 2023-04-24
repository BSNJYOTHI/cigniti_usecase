# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the main.py and app.py into the container at /app
COPY main.py .
COPY app.py .

# Expose port 8000 for FastAPI and 8501 for Streamlit
EXPOSE 8000
EXPOSE 8501

# Run FastAPI and Streamlit using uvicorn and streamlit command respectively
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--reload"] && ["streamlit", "run", "app.py", "--server.port", "8501"]
