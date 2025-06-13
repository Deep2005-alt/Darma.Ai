import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20  # Increased for better learning
MODEL_PATH = 'skin_disease_model.pth'
DATASET_PATH = 'skin_dataset'
CLASSES = [
    'actinic_keratosis', 'basal_cell_carcinoma', 'benign_keratosis',
    'skin_dataset/dermatofibroma', 'melanoma',
    'melanocytic_nevi', 'skin_dataset/vascular_lesions'
]

# ✅ Data augmentation + normalization
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Load dataset
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Build model
model = models.mobilenet_v2(pretrained=True)

# ✅ Unfreeze last few layers for fine-tuning
for param in model.features[:-5].parameters():
    param.requires_grad = False

# ✅ Replace classifier (No softmax)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Linear(128, len(CLASSES))
)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Continue training from saved model if exists
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Loaded existing model weights. Continuing training...")
else:
    print("🆕 Starting training from scratch...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Train loop
train_acc, val_acc = [], []
train_loss, val_loss = [], []

for epoch in range(EPOCHS):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_train_acc = correct / total
    epoch_train_loss = running_loss / len(train_loader)

    model.eval()
    val_total, val_correct, val_running_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    epoch_val_acc = val_correct / val_total
    epoch_val_loss = val_running_loss / len(val_loader)

    train_acc.append(epoch_train_acc)
    val_acc.append(epoch_val_acc)
    train_loss.append(epoch_train_loss)
    val_loss.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')

plt.savefig('training_history.png')
