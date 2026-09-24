FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV KMP_DUPLICATE_LIB_OK=TRUE

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
