# =============================================================================
# Production Dockerfile for Esports Sponsorship Media Value Forecasting Engine
# 100% Local-Compatible & Cloud-Agnostic (No AWS Dependencies)
# =============================================================================
FROM python:3.12-slim

# Set environment variables for Python in container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set container working directory
WORKDIR /app

# Install runtime libraries for XGBoost and the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    grep -vi "^xgboost" requirements.txt > /tmp/requirements-no-xgboost.txt && \
    pip install --no-cache-dir -r /tmp/requirements-no-xgboost.txt && \
    pip install --no-cache-dir --no-deps "xgboost>=2.0.0"

# Copy application codebase
COPY src/ /app/src/
COPY data/ /app/data/
COPY models/ /app/models/
COPY reports/ /app/reports/
COPY app.py /app/app.py
COPY esports_player_performance_tournament_analytics.csv /app/esports_player_performance_tournament_analytics.csv

# Expose Streamlit default web port
EXPOSE 8501

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
