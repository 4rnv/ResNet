import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.datasets import ImageFolder
from PIL import Image

# Define the data transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your dataset of images and labels using ImageFolder
dataset = ImageFolder('images', transform=transform)

# Print dataset information
print(f'Number of classes: {len(dataset.classes)}')
print(f'Classes: {dataset.classes}')
print(f'Number of images: {len(dataset)}')

# Create the data loader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load a pre-trained ResNet model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

# Evaluate the model
model.eval()

# Load a new image for prediction
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Test image (dog)
image_path = 'd108.png'
# Test image (cat)
#image_path = '73.png'

image = load_and_preprocess_image(image_path)

# Run the model
with torch.no_grad():
    outputs = model(image)

# Print raw outputs
print("Raw outputs:", outputs)

# Get the predicted label
predicted_label = torch.argmax(outputs, dim=1)
predicted_class_name = dataset.classes[predicted_label]
print(f'Predicted label: {predicted_label} ({predicted_class_name})')

# Check the probabilities
probabilities = torch.softmax(outputs, dim=1)
print("Probabilities:", probabilities)