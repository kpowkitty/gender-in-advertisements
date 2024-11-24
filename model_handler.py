import torch
from dataloader_wrapper import DataLoaderDeviceWrapper
import matplotlib.pyplot as plt

class ModelHandler:
    """Handles predictions and evaluations for a model."""
    def __init__(self, model, dataset, device):
        self.model = model.to(device)  # Correctly move the model to the device
        self.dataset = dataset
        self.device = device

    def predict_image(self, img):
        """Predict the class of a single image."""
        self.model.eval()
        xb = self.to_device(img.unsqueeze(0))  # Move the image to the device
        yb = self.model(xb)  # Get predictions
        _, preds = torch.max(yb, dim=1)  # Pick the class with the highest probability
        return self.dataset.classes[preds[0].item()]

    def evaluate_model(self, test_loader):
        """Evaluate the model on a test dataset."""
        self.model.eval()
        with torch.no_grad():
            outputs = [self.model.validation_step(batch) for batch in test_loader]
        return self.model.validation_epoch_end(outputs)

    def to_device(self, data):
        """Move tensor(s) to the chosen device."""
        if isinstance(data, (list, tuple)):
            return [x.to(self.device, non_blocking=True) for x in data]
        return data.to(self.device, non_blocking=True)