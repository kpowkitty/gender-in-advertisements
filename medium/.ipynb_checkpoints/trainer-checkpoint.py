import torch

class Trainer:
    """Handles the training process of the model."""

    def __init__(self, model, opt_func=torch.optim.SGD):
        """
        Args:
            model: The model to train.
            opt_func: The optimizer function (default is SGD).
        """
        self.model = model
        self.opt_func = opt_func

    def fit(self, epochs, lr, train_loader, val_loader):
        """Train the model for a given number of epochs."""
        history = []
        optimizer = self.opt_func(self.model.parameters(), lr)  # Initialize optimizer

        evaluator = Evaluator()  # Create an evaluator instance

        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            train_losses = []
            for batch in train_loader:
                loss = self.model.training_step(batch)  # Perform training step
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validation Phase
            result = evaluator.evaluate(self.model, val_loader)  # Get validation results
            result['train_loss'] = torch.stack(train_losses).mean().item()  # Calculate average train loss
            self.model.epoch_end(epoch, result)  # Log epoch results
            history.append(result)  # Store history

        return history
