# ---- base image ----
    FROM python:3.11-slim

    # ---- system settings ----
    ENV PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PIP_DEFAULT_TIMEOUT=100
    
    # ---- install OS packages  --------------------------------------------
      RUN apt-get update \
      && apt-get install -y --no-install-recommends \
         tesseract-ocr \
         libtesseract-dev \
         poppler-utils \          
         libgl1 \    
      && rm -rf /var/lib/apt/lists/*
    # ---- Python deps ----------------------------------------------------------
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    COPY . .
    
    # ---- default command ----
    CMD ["bash"]
    