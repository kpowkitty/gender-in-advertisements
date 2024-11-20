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

print(os.listdir(database.dir)[:10])
