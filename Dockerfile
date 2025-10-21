FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl

# System packages for fonts and OpenMP (LightGBM/sklearn), keep image minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    fontconfig \
    fonts-dejavu-core \
    fonts-noto-cjk \
    fonts-nanum \
  && fc-cache -f -v \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY dashboard/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app source (includes data under dashboard/data)
COPY dashboard /app/dashboard

EXPOSE 8080
WORKDIR /app/dashboard

# Run via Uvicorn with proxy headers enabled (required behind Fly proxy)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
