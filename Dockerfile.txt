# Step 1: Use Python 3.9 as base image
FROM python:3.9

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy all files to container
COPY . /app

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose FastAPI default port
EXPOSE 8000

# Step 6: Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
