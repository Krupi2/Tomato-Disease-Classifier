from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.templating import Jinja2Templates
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="Tomato Leaf Disease API",
    description="API for classifying tomato leaf diseases using an optimized ONNX model.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------
# 1. Ładowanie sesji ONNX
# ---------------------------------------------------------
try:
    ort_session = ort.InferenceSession("tomato_model.onnx")
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    raise RuntimeError(
        f"Nie udało się załadować modelu ONNX. Upewnij się, że plik tomato_model.onnx istnieje. Błąd: {e}")

# ---------------------------------------------------------
# 2. Mapowanie klas posortowane alfabetycznie
# ---------------------------------------------------------
CLASS_NAMES = [
    "Bacterial_spot",  # Indeks 0
    "Early_blight",  # Indeks 1
    "Late_blight",  # Indeks 2
    "Leaf_Mold",  # Indeks 3
    "Septoria_leaf_spot",  # Indeks 4
    "Spider_mites Two-spotted_spider_mite",  # Indeks 5
    "Target_Spot",  # Indeks 6
    "Tomato_Yellow_Leaf_Curl_Virus",  # Indeks 7
    "Tomato_mosaic_virus",  # Indeks 8
    "healthy",  # Indeks 9
    "powdery_mildew"  # Indeks 10
]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))

        img_data = np.array(image).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std

        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        return img_data
    except Exception as e:
        raise ValueError(f"Błąd przetwarzania obrazu: {e}")


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Plik musi być obrazem!")

    try:
        image_bytes = await file.read()
        input_data = preprocess_image(image_bytes)

        ort_inputs = {input_name: input_data}
        ort_out = ort_session.run(None, ort_inputs)[0]

        predicted_idx = int(np.argmax(ort_out[0]))
        confidence_scores = np.exp(ort_out[0]) / np.sum(np.exp(ort_out[0]))
        confidence = float(confidence_scores[predicted_idx])

        predicted_class_name = CLASS_NAMES[predicted_idx] if predicted_idx < len(
            CLASS_NAMES) else f"Unknown_Class_{predicted_idx}"

        return {
            "filename": file.filename,
            "predicted_class_id": predicted_idx,
            "disease_name": predicted_class_name,
            "confidence": f"{confidence * 100:.2f}%"
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wewnętrzny błąd serwera: {e}")