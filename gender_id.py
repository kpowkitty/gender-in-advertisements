import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

from database import Database
from class_finder import ClassFinder
from dataloader_wrapper import DataLoaderDeviceWrapper
from dataset_handler import DatasetHandler
from device import Device
from evaluator import Evaluator
from image_classification import ImageClassificationBase
from kernel import Conv2D
from model_handler import ModelHandler
from trainer import Trainer
from visualizer import Visualizer

# Cell 1 - Initialize dataset and dataloader, set transform, and show_example
DATA_DIR = './menwomen-classification/traindata/traindata'

# Note: if you resize the image, you must change class finder's matrix
# multiplication to match the new expected dimension
t = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
database = Database("menwomen-classification",
                    DATA_DIR,
                    ImageFolder(DATA_DIR, transform=t))
val_size = 500
train_size = len(database.dataset) - val_size
train_ds, val_ds = random_split(database.dataset, [train_size, val_size])

# # of images
batch_size=16
train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                      num_workers=2,
                      pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)

database.show_example(1010)

# Cell 2 - show_batch
database.show_batch(train_dl)

# Cell 3 - Testing the kernel
# Define a sample kernel (e.g., a simple edge detection kernel)
kernel = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

# Define a sample image
image = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# Initialize the Conv2D class with the kernel
conv = Conv2D(kernel)

# Apply the kernel to the image
output = conv.apply_kernel(image)

print(output)

# Cell 4 - Initialization and testing
# Initialize the model
model = ClassFinder()

# Example of a training batch (replace this with actual DataLoader output)
# images: Tensor of shape [batch_size, channels, height, width]
# labels: Tensor of shape [batch_size]
images = torch.randn(16, 3, 128, 128)  # Example image tensor (32 images, 3 channels, 32x32 pixels)
labels = torch.randint(0, 10, (16,))  # Example labels tensor (32 labels, for 10 classes)

# Run a training step
batch = (images, labels)
loss = model.training_step(batch)
print(f"Training loss: {loss.item()}")

# Run a validation step
val_results = model.validation_step(batch)
print(f"Validation loss: {val_results['val_loss'].item()}, Validation accuracy: {val_results['val_acc'].item()}")

model

# Cell 5 - Training the model
# Example of usage
device = Device.get_default_device()
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
train_dl_device = DataLoaderDeviceWrapper(train_dl, device)

for data in train_dl_device:
    # Now each batch is moved to the correct device
    images, labels = data
    # Further processing can be done here to fit your needs

trainer = Trainer(model, train_loader=train_dl, val_loader=val_dl, num_epochs=10, lr=0.001, opt_func=torch.optim.Adam)
history = trainer.fit()

# Cell 6 - Plotting the accuracy vs. epochs
Visualizer.plot_accuracies(history)

# Cell 7 - Plotting the losses
Visualizer.plot_losses(history)

# Cell 8 - Saving the model
dataset_handler = DatasetHandler(DATA_DIR)
test_dataset = dataset_handler.get_dataset()

torch.save(model.state_dict(), 'men_women.pth')
model2 = to_device(class_finder(), device)
model2.load_state_dict(torch.load('men_women.pth'))

## Change the dataset & use pre-trained model from above
# Define the path to your new dataset
# DATA_DIR = './ads_in_80s'

# Define your transformations (e.g., resizing and converting to tensor)
# t = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# Load the dataset using ImageFolder
# new_dataset = ImageFolder(DATA_DIR, transform=t)

# Create a DataLoader for the new dataset
# batch_size = 32
# new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Load the pre-trained model (model2)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Ensure the model is on the correct device
# model2.eval()  # Set model to evaluation mode

# Example usage: Iterate through the new dataset and make predictions
# for images, labels in new_loader:
#     images = to_device(images, device)  # Move images to the correct device
#     outputs = model2(images)  # Forward pass through the model
#     _, predictions = torch.max(outputs, dim=1)  # Get predicted class indices

#     # Print predictions (you can process them further as needed)
#     print(f"Predicted classes: {predictions}")
#     print(f"True labels: {labels}")
