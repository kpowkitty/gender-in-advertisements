import matplotlib.pyplot as plt

class Visualizer:
    """Handles the visualization of training and validation metrics."""

    @staticmethod
    def plot_accuracies(history):
        """
        Plot validation accuracies across epochs.
        
        Args:
            history: List of dictionaries containing epoch results.
        """
        accuracies = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. No. of Epochs')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_losses(history):
        """
        Plot training and validation losses across epochs.
        
        Args:
            history: List of dictionaries containing epoch results.
        """
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx', label='Training Loss')
        plt.plot(val_losses, '-rx', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of Epochs')
        plt.show()