import torch

class Evaluator:
    """Handles the evaluation of the model."""
    
    @torch.no_grad()
    def evaluate(self, model, val_loader):
        """Evaluate the model on the validation set."""
        model.eval()  # Set model to evaluation mode
        outputs = [model.validation_step(batch) for batch in val_loader]  # Perform validation
        return model.validation_epoch_end(outputs)  # Return aggregated resultsmodel = to_device(class_finder(), device)evaluate(model, val_dl)