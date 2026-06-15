# 🍅 Tomato Doctor AI - Klasyfikator Chorób Liści Pomidora (MLOps Pipeline)

Kompleksowy projekt typu end-to-end obejmujący trenowanie głębokich sieci neuronowych (Transfer Learning), optymalizację czasu inferencji (ONNX) oraz wdrożenie modelu w architekturze mikrousługowej z wykorzystaniem FastAPI, nowoczesnego interfejsu webowego i Dockera.

## 1. Cel Projektu
Głównym założeniem projektu było stworzenie wydajnego klasyfikatora obrazów zdolnego do rozpoznawania 10 różnych chorób liści pomidora (oraz klasy liści zdrowych) na podstawie zdjęć. Projekt miał na celu nie tylko samo wytrenowanie modelu, ale również jego optymalizację pod kątem środowisk produkcyjnych (zmniejszenie opóźnień) i zamknięcie w gotowym do wdrożenia kontenerze.

## 2. Zbiór Danych
Do nauki modelu wykorzystano publiczny zbiór danych z platformy Kaggle: **[ashishmotwani/tomato](https://www.kaggle.com/datasets/ashishmotwani/tomato)**.
* **Rozmiar:** Około 32 000 obrazów (25 851 w zbiorze treningowym, 6 683 w walidacyjnym).
* **Klasy (11):** `Bacterial_spot`, `Early_blight`, `Late_blight`, `Leaf_Mold`, `Septoria_leaf_spot`, `Spider_mites`, `Target_Spot`, `Tomato_Yellow_Leaf_Curl_Virus`, `Tomato_mosaic_virus`, `powdery_mildew` oraz `healthy`.

## 3. Proces Uczenia i Transfer Learning
W projekcie wykorzystano architekturę **ResNet18**. Proces uczenia podzielono na dwa etapy:
1. **Feature Extraction:** Zamrożenie wag bazowych modelu i trenowanie wyłącznie nowej warstwy klasyfikatora. Pozwoliło to na szybkie osiągnięcie ~79% dokładności w zaledwie kilka epok (z wykorzystaniem technologii DirectML).
2. **Fine-Tuning:** Odmrożenie wszystkich warstw sieci i wykorzystanie optymalizatora SGD (Stochastic Gradient Descent) z niskim współczynnikiem uczenia (Learning Rate = 0.0001).
   
**Wyniki końcowe walidacji:**
* Ostateczna dokładność modelu (Accuracy): **87.57%**
* Czas Fine-Tuningu (5 epok): **~13 minut**

## 4. Optymalizacja i Eksport do ONNX
Aby uniezależnić aplikację od biblioteki PyTorch i znacząco obniżyć czas predykcji, wytrenowane wagi (`.pth`) zostały wyeksportowane do uniwersalnego formatu **ONNX** (Open Neural Network Exchange).

Zgodność predykcji PyTorch vs. ONNX Runtime została zweryfikowana - uzyskano **100% zgodności** wyników.

**Testy wydajności (Średni czas inferencji ze 200 prób na procesorze CPU):**
* **PyTorch CPU:** ~21.34 ms / obraz
* **ONNX Runtime CPU:** ~9.83 ms / obraz

> **Wniosek:** Format ONNX przyspieszył działanie modelu **2.17-krotnie**, bez żadnej straty na jakości predykcji, co czyni go idealnym wyborem do wdrożeń chmurowych opartych o tańsze serwery CPU.

## 5. Architektura Aplikacji
Aplikacja serwerowa została napisana w języku Python z użyciem:
* **FastAPI:** Szybki i lekki framework do budowy API.
* **ONNX Runtime:** Silnik inferencyjny modelu.
* **Jinja2 & Tailwind CSS:** Wygenerowanie nowoczesnego, responsywnego interfejsu graficznego (Drag & Drop) bez konieczności tworzenia osobnego projektu frontendowego.
* **Docker:** Całość działa na obrazie `python:3.11-slim`.

## 6. Struktura Projektu (Opis Plików)

W repozytorium znajdują się następujące pliki, z których każdy pełni określoną rolę w cyklu życia modelu:

* **Trenowanie i Optymalizacja:**
  * `train.py` / `train_finetune.py` - Skrypty w języku Python oparte na bibliotece PyTorch, wykorzystane do przeprowadzenia procesu Transfer Learningu oraz Fine-Tuningu modelu na bazowym zbiorze danych.
  * `onnx_pipeline.py` - Skrypt odpowiadający za eksport wytrenowanych wag modelu do formatu `.onnx`. Dodatkowo weryfikuje zgodność wyników (PyTorch vs ONNX) i mierzy wydajność (czas inferencji na CPU).
  * `tomato_model.onnx` - Gotowy, zoptymalizowany model sieci neuronowej w uniwersalnym formacie ONNX, wykorzystywany przez aplikację webową.

* **Aplikacja Webowa:**
  * `main.py` - Główny plik serwera napisany z użyciem FastAPI. Definiuje endpointy API, wczytuje sesję `onnxruntime` i przetwarza nadsyłane przez użytkowników obrazy.
  * `templates/index.html` - Plik frontendowy oparty na szablonach Jinja2. Zawiera responsywny interfejs użytkownika zrealizowany przy pomocy Tailwind CSS z napisaną w JavaScript obsługą funkcji Drag & Drop.

* **Wdrożenie (Docker):**
  * `Dockerfile` - Instrukcja budowy kontenera. Definiuje bazowy system (Python 3.11-slim) oraz kroki niezbędne do uruchomienia aplikacji w izolowanym środowisku.
  * `.dockerignore` - Lista ignorowanych plików (takich jak wirtualne środowiska `.venv` czy surowe pliki `.pth`), co pozwala zachować mały rozmiar obrazu platformy Docker.
  * `requirements.txt` - Zablokowana lista niezbędnych zależności i bibliotek w języku Python.
  * `build.bat` & `run.bat` - Skrypty automatyzujące budowanie i uruchamianie kontenera dla systemu Windows.
  * `build.sh` & `run.sh` - Skrypty automatyzujące budowanie i uruchamianie kontenera dla środowisk Linux / macOS.

## Instrukcja Uruchomienia (Docker)

Projekt jest całkowicie zautomatyzowany. Nie musisz instalować Pythona ani bibliotek na swoim komputerze, wystarczy samo narzędzie **Docker**.

### Dla systemu Windows:
1. Sklonuj lub pobierz to repozytorium na swój dysk.
2. Upewnij się, że masz uruchomiony program **Docker Desktop**.
3. Wejdź do folderu z projektem i dwukrotnie kliknij plik `build.bat`, aby zbudować kontener.
4. Po udanym budowaniu dwukrotnie kliknij plik `run.bat`, aby wystartować serwer.

### Dla systemów Linux / macOS:
1. Otwórz terminal wewnątrz folderu projektu.
2. Nadaj uprawnienia do uruchamiania skryptów:
   ```bash
   chmod +x build.sh run.sh