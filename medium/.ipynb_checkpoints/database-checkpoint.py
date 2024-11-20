from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

class Database:
    def __init__(self, name, dir, dataset):
        self._name = name
        self._dir = dir
        self._dataset = dataset

    @property
    def name(self):
        """The name property."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def dir(self):
        """The data_dir property."""
        return self._dir

    @dir.setter
    def dir(self, value):
        self._dir = value

    @property
    def dataset(self):
        """The dataset property."""
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def show_example(self, idx):
        img, label = self.dataset[idx]
        print('Label: ', self.dataset.classes[label], f"({label})")
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

    def show_batch(self, dl):
        #for images, labels in dl:
        #    fig, ax = plt.subplots(figsize=(12, 6))
        #    ax.set_xticks([]); ax.set_yticks([])
        #    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        #    plt.show()
        #    # break to display only one batch
        #    # remove to display all
        #    break
        try:
            images, labels = next(iter(dl))
            img_grid = make_grid(images, nrow=16).permute(1, 2, 0)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img_grid)
            plt.show()
        except StopIteration:
            print("The DataLoader is empty.")
        except Exception as e:
            print(f"An error occurred: {e}")
