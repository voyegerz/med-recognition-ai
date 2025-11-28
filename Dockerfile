# Use an official lightweight Python image
FROM python:3.11-slim

# 1. Install System Dependencies (Tesseract & OpenGL for OpenCV)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy files
COPY . .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose port (Render sets the PORT env var, we default to 10000)
ENV PORT=10000
EXPOSE 10000

# 6. Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]