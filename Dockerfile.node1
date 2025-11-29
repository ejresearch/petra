FROM python:3.11-slim

WORKDIR /app

COPY node1_requirements.txt .
RUN pip install --no-cache-dir -r node1_requirements.txt

COPY src/ src/

EXPOSE 8001

CMD ["python", "src/node1_backend.py"]
