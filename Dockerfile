# 1. Obraz Pythona 3.11
FROM python:3.11-slim

# 2. Katalog roboczy wewnątrz kontenera
WORKDIR /app

# 3. Plik z listą bibliotek do python'a
COPY requirements.txt .

# 4. Instalacja bibliotek
RUN pip install --no-cache-dir -r requirements.txt

# 5. Kod aplikacji, wyeksportowany model ONNX oraz szablony HTML
COPY main.py .
COPY tomato_model.onnx .
COPY templates/ ./templates/

# 6. Port na którym aplikacja będzie nasłuchiwać
EXPOSE 8000

# 7. Komenda, która uruchamia serwer FastAPI po starcie kontenera
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]