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
DATA_DIR = r"C:\Users\Krupiz\.cache\kagglehub\datasets\ashishmotwani\tomato\versions\1"

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

# ---------------------------------------------------------
# 2. Przygotowanie Danych
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

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
    val_folder_name: datasets.ImageFolder(os.path.join(DATA_DIR, val_folder_name), data_transforms[val_folder_name])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    val_folder_name: DataLoader(image_datasets[val_folder_name], batch_size=32, shuffle=False)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', val_folder_name]}
num_classes = len(image_datasets['train'].classes)

# ---------------------------------------------------------
# 3. Model - Wczytanie dotychczasowego stanu
# ---------------------------------------------------------
print("\nŁadowanie bazowego modelu do Fine-Tuningu...")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Ładujemy model, który został wytrenowany w poprzednim kroku (train.py)
if os.path.exists("custom_resnet18_11_classes.pth"):
    model.load_state_dict(torch.load("custom_resnet18_11_classes.pth", map_location='cpu'))
    print("-> Wczytano wagi dotychczasowego modelu (79% Acc).")

# Odblokowujemy wszystkie warstwy do trenowania
for param in model.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# ---------------------------------------------------------
# 4. Pętla Fine-Tuningu
# ---------------------------------------------------------
num_epochs = 5
best_acc = 0.7906 #należy ustawić na dokładność najlepszego modelu, który uzyskaliśmy wcześniej w wykonaniu train.py
start_time = time.time()

for epoch in range(num_epochs):
    print(f'\nFine-tuning - Epoka {epoch+1}/{num_epochs}')
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
            print(f"!!! Nowy najlepszy model zapisany z dokładnością: {best_acc:.4f} !!!")

time_elapsed = time.time() - start_time
print(f'\nFine-tuning ukończony w czasie {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Ostateczna najlepsza dokładność: {best_acc:4f}')