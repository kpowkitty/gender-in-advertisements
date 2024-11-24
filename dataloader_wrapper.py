import torch
from device import Device
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader 

class DataLoaderDeviceWrapper:
    """Wraps a dataloader to move data to a device."""
    
    def __init__(self, dl, device):
        """
        Args:
            dl: The DataLoader to wrap.
            device: The device to move data to (e.g., 'cuda' or 'cpu').
        """
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to the specified device."""
        for b in self.dl:
            yield self.to_device(b)

    def __len__(self):
        """Number of batches."""
        return len(self.dl)

    def to_device(self, data):
        """Move tensor(s) to the chosen device."""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)
