# Use a professional, lightweight Python image
FROM python:3.12-slim

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and Image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency file first for caching
COPY pyproject.toml .

# Install UV and dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r pyproject.toml

# Copy only the strictly necessary serving files
COPY model.py config.py ./
COPY checkpoints/best_model.pth ./checkpoints/best_model.pth
COPY serving/ ./serving/

# Expose the port
EXPOSE 8000

# Start the production server
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
