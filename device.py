import torch

class Device:
    """Handles device-related operations."""
    
    @staticmethod
    def get_default_device():
        """Pick GPU if available, else CPU."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')