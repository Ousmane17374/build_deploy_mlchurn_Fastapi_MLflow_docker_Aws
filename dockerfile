# 1. Use the official lightweight Python base image
FROM python:3.12-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project into the image
COPY . .

# make "src" importable + show logs in real-time
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# 6. Expose FastAPI port
EXPOSE 8000

# 7. Run the FastAPI app using uvicorn
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
