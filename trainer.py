import torch
from evaluator import Evaluator

class Trainer:
    """Handles the training process of the model."""
    
    def __init__(self, model, train_loader, val_loader, num_epochs=10, lr=0.001, opt_func=torch.optim.Adam):
        """
        Args:
            model: The model to train.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            num_epochs: Number of epochs to train.
            lr: Learning rate.
            opt_func: The optimizer function (default is Adam).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.opt_func = opt_func
        self.history = []

    def fit(self):
        """Train the model for the specified number of epochs."""
        optimizer = self.opt_func(self.model.parameters(), self.lr)  # Initialize optimizer
        evaluator = Evaluator()  # Create an evaluator instance

        for epoch in range(self.num_epochs):
            # Training Phase
            self.model.train()
            train_losses = []
            for batch in self.train_loader:
                loss = self.model.training_step(batch)  # Perform training step
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validation Phase
            result = evaluator.evaluate(self.model, self.val_loader)  # Get validation results
            result['train_loss'] = torch.stack(train_losses).mean().item()  # Calculate average train loss
            self.model.epoch_end(epoch, result)  # Log epoch results
            self.history.append(result)  # Store history

        return self.history
