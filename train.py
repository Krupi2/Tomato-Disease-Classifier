import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import torch_directml

# ---------------------------------------------------------
# 1. Konfiguracja ścieżki i urządzenia na którym będzie trenowany model
# ---------------------------------------------------------
DATA_DIR = r"C:\Users\Krupiz\.cache\kagglehub\datasets\ashishmotwani\tomato\versions\1" # <-- Upewnij się, że ścieżka jest poprawna

# Sprawdzenie dostępności DirectML i ustawienie urządzenia
if torch_directml.is_available():
    device = torch_directml.device()
    print(f"DirectML dostępne. Używam: {device}")
else:
    device = torch.device("cpu")
    print("DirectML niedostępne. Używam CPU.")

# Automatyczne wykrywanie nazwy folderu walidacyjnego
val_folder_name = None
for name in ['val', 'valid', 'test']:
    if os.path.exists(os.path.join(DATA_DIR, name)):
        val_folder_name = name
        break

if not val_folder_name:
    raise FileNotFoundError(f"Nie znaleziono folderu walidacyjnego w ścieżce {DATA_DIR}!")

# ---------------------------------------------------------
# 2. Przygotowanie danych
# ---------------------------------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_folder_name: transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

print(f"Wczytywanie danych z: {DATA_DIR}...")
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
    val_folder_name: datasets.ImageFolder(os.path.join(DATA_DIR, val_folder_name), data_transforms[val_folder_name])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    val_folder_name: DataLoader(image_datasets[val_folder_name], batch_size=32, shuffle=False)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', val_folder_name]}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"Rozmiar treningowy: {dataset_sizes['train']} | Rozmiar walidacyjny: {dataset_sizes[val_folder_name]}")

# ---------------------------------------------------------
# 3. Model i Transfer Learning
# ---------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ---------------------------------------------------------
# 4. Pętla ucząca
# ---------------------------------------------------------
num_epochs = 8
best_acc = 0.0
start_time = time.time()

for epoch in range(num_epochs):
    print(f'\nEpoka {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', val_folder_name]:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data).to("cpu")

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase.capitalize()} -> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == val_folder_name and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.to("cpu").state_dict(), "custom_resnet18_11_classes.pth")
            model = model.to(device)

time_elapsed = time.time() - start_time
print(f'\nTrening ukończony w czasie {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Najlepsza dokładność: {best_acc:4f}')
print("Zapisano model jako 'custom_resnet18_11_classes.pth'")