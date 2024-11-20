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

DATA_DIR = './menwomen-classification/traindata/traindata'
t = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
database = Database("menwomen-classification",
                    DATA_DIR,
                    ImageFolder(DATA_DIR, transform=t))
val_size = 500
train_size = len(database.dataset) - val_size
train_ds, val_ds = random_split(database.dataset, [train_size, val_size])

batch_size=128
train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                      num_workers=2,
                      pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)

database.show_example(1010)
database.show_batch(train_dl)

# Define a sample kernel (e.g., a simple edge detection kernel)
kernel = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

# Define a sample image
image = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# Initialize the Conv2D class with the kernel
conv = Conv2D(kernel)

# Apply the kernel to the image
output = conv.apply_kernel(image)

print(output)

# Initialize the model
model = ClassFinder()

# Example of a training batch (replace this with actual DataLoader output)
# images: Tensor of shape [batch_size, channels, height, width]
# labels: Tensor of shape [batch_size]
images = torch.randn(32, 3, 32, 32)  # Example image tensor (32 images, 3 channels, 32x32 pixels)
labels = torch.randint(0, 10, (32,))  # Example labels tensor (32 labels, for 10 classes)

# Run a training step
batch = (images, labels)
loss = model.training_step(batch)
print(f"Training loss: {loss.item()}")

# Run a validation step
val_results = model.validation_step(batch)
print(f"Validation loss: {val_results['val_loss'].item()}, Validation accuracy: {val_results['val_acc'].item()}")

model

# Example of usage
device = Device.get_default_device()
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
train_dl_device = DataLoaderDeviceWrapper(train_dl, device)

for data in train_dl_device:
    # Now each batch is moved to the correct device
    images, labels = data
    # Further processing...

trainer = Trainer(model, train_loader=train_dl, val_loader=val_dl, num_epochs=10, lr=0.001, opt_func=torch.optim.Adam)
history = trainer.fit()

Visualizer.plot_accuracies(history)
Visualizer.plot_losses(history)

dataset_handler = DatasetHandler(DATA_DIR)
test_dataset = dataset_handler.get_dataset()

model_handler = ModelHandler(model=model, dataset=test_dataset, device=device)

# Predict a single image
img, label = test_dataset[1002]
plt.imshow(img.permute(1, 2, 0))
print(
    f"Label: {test_dataset.classes[label]}, "
    f"Predicted: {model_handler.predict_image(img)}"
)

# Evaluate the model on the test dataset
test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size=32), device=device)  # Ensure batch_size is defined
result = model_handler.evaluate_model(test_loader)
print(result)

torch.save(model.state_dict(), ‘men_women.pth’)
model2 = to_device(class_finder(), device)
model2.load_state_dict(torch.load(‘men_women.pth’))
