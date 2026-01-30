# 1. Use the official lightweight Python base image
FROM python:3.12-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Install system dependencies required by LightGBM (OpenMP)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 5. Install python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire project into the image
COPY . .

# 7. Make "src" importable + show logs in real-time
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# 8. Expose FastAPI port
EXPOSE 8000

# 9. Run the FastAPI app using uvicorn
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
