#!/bin/bash
echo "========================================="
echo "Uruchamianie kontenera tomato-ai..."
echo "Aplikacja będzie dostępna pod adresem: http://127.0.0.1:8000"
echo "Naciśnij CTRL+C, aby zatrzymać serwer."
echo "========================================="
docker run -p 8000:8000 tomato-ai