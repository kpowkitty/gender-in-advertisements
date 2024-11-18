from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd

Step 1: Load the pre-trained model (e.g., ResNet18)
model = models.resnet18(weights='IMAGENET1KV1')
model.eval()  # Set the model to evaluation mode

Step 2: Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 (standard for most pre-trained models)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

Step 3: Load the images from a directory and make predictions
image_dir = "/home/bee/Pictures/ads90s"  # Path to your image directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))]

Step 4: Make predictions
predictions = []
with torch.no_grad():  # No need to track gradients for inference
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        output = model(image)  # Forward pass through the model
        , predicted_class = torch.max(output, 1)  # Get the class with the highest probability

        # Convert predicted class index to human-readable label
        predicted_class = predicted_class.item()
        predictions.append((image_file, predicted_class))

Step 5: Save predictions to a spreadsheet
df = pd.DataFrame(predictions, columns=["Image", "Predicted Class"])
df.to_excel("image_predictions.xlsx", index=False)

print("Predictions saved to image_predictions.xlsx")
