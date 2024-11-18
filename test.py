import torch
import numpy as np
import pandas as pd
import torchvision.io as io
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Step 1: Load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for pre-trained models
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
dataset = ImageFolder("~/Pictures/ads90s", transform=transform)
dataloader = DataLoader(dataset, batch_size=32)

# Step 2: Load a pre-trained model (e.g., ResNet18)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Assuming binary classification (man/woman)
model.load_state_dict(torch.load("path_to_finetuned_model.pth"))  # Load fine-tuned weights
model.eval()

# Step 3: Predict and count occurrences
counts = {"man": 0, "woman": 0}
classes = ["man", "woman"]  # Ensure these match your model's output labels

with torch.no_grad():
    for images, _ in dataloader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)  # Get class with highest probability
        for pred in predictions:
            counts[classes[pred]] += 1

# Step 4: Save to spreadsheet
df = pd.DataFrame([counts])
df.to_excel("gender_counts.xlsx", index=False)

print("Counts saved to gender_counts.xlsx")

