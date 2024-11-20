from torchvision.datasets import ImageFolder
from torchvision import transforms

class DatasetHandler:
    """Manages dataset creation and preprocessing."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        self.dataset = ImageFolder(data_dir, transform=self.transform)

    def get_dataset(self):
        return self.dataset