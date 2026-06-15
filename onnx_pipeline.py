import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import time
from torchvision import models

# ---------------------------------------------------------
# Eksport do ONNX
# ---------------------------------------------------------
print("--- Ładowanie modelu i Eksport do ONNX ---")
num_classes = 11
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Ładujemy model, który przeszedł fine-tuning (train_finetune.py)
model.load_state_dict(torch.load("custom_resnet18_11_classes.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Sztuczne zdjęcie do eksportu (1 obraz, 3 kanały, 224x224 piksele)
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "tomato_model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Sukces! Model zapisany jako: {onnx_path}\n")

# ---------------------------------------------------------
# Zgodność Predykcji
# ---------------------------------------------------------
print("--- Sprawdzanie zgodności predykcji ---")

# Wynik PyTorch
with torch.no_grad():
    torch_out = model(dummy_input).numpy()

# Wynik ONNX Runtime
ort_session = ort.InferenceSession(onnx_path)
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_out = ort_session.run(None, ort_inputs)[0]

# Porównanie z tolerancją na zaokrąglenia
try:
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
    print("Predykcje są w 100% zgodne!\n")
except AssertionError as e:
    print("Uwaga! Różnice w predykcjach:", e)

# ---------------------------------------------------------
# Pomiar Czasu Inferencji (CPU)
# ---------------------------------------------------------
print("--- Pomiar czasu inferencji (Uśredniony z 200 prób) ---")
iterations = 200

for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)
    _ = ort_session.run(None, ort_inputs)

# Test PyTorch CPU
start_pytorch = time.perf_counter()
for _ in range(iterations):
    with torch.no_grad():
        _ = model(dummy_input)
end_pytorch = time.perf_counter()
pytorch_avg_time = ((end_pytorch - start_pytorch) / iterations) * 1000

# Test ONNX Runtime CPU
start_onnx = time.perf_counter()
for _ in range(iterations):
    _ = ort_session.run(None, ort_inputs)
end_onnx = time.perf_counter()
onnx_avg_time = ((end_onnx - start_onnx) / iterations) * 1000

print(f"Średni czas PyTorch (CPU): {pytorch_avg_time:.2f} ms na zdjęcie")
print(f"Średni czas ONNX Runtime (CPU): {onnx_avg_time:.2f} ms na zdjęcie")

speedup = pytorch_avg_time / onnx_avg_time
print(f"-> Zysk: ONNX jest {speedup:.2f}x szybszy!")