import torch
from dataloader_wrapper import DataLoaderDeviceWrapper
import matplotlib.pyplot as plt

class ModelHandler:
    """Handles predictions and evaluations for a model."""
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model = DeviceDataLoader.to_device(self, model)

    def predict_image(self, img):
        """Predict the class of a single image."""
        self.model.eval()
        xb = DeviceDataLoader.to_device(self, img.unsqueeze(0))
        yb = self.model(xb)  # Get predictions
        _, preds = torch.max(yb, dim=1)  # Pick the class with highest probability
        return self.dataset.classes[preds[0].item()]

    def evaluate_model(self, test_loader):
        """Evaluate the model on a test dataset."""
        self.model.eval()
        with torch.no_grad():
            outputs = [self.model.validation_step(batch) for batch in test_loader]
        return self.model.validation_epoch_end(outputs)