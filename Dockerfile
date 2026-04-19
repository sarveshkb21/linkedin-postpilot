FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
