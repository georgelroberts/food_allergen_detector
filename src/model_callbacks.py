from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import wandb

class PlotSamplesCallback(Callback):
    def __init__(self, wandb_logger: WandbLogger):
        super().__init__()
        self.wandb_logger = wandb_logger
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        if batch_idx == 0:
            n = 16
            try:
                x, y, _, _ = batch
            except ValueError:
                x, y, _, _ = batch[0]
            images = [img for img in x[:n]]
            captions = [f"Ground Truth: {bool(y_i)} - Prediction: {y_pred.numpy()[0]}" 
                for y_i, y_pred in zip(y[:n], (outputs[0] > 0)[:n])]
            
            self.wandb_logger.log_image(
                key="sample_images", 
                images=images, 
                caption=captions)


class PlotIncorrectSamplesCallback(Callback):
    def __init__(self, wandb_logger: WandbLogger, no_to_plot: int):
        super().__init__()
        self.wandb_logger = wandb_logger
        self.no_to_plot = no_to_plot
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        if batch_idx != trainer.num_val_batches[0] - 1:
            # Do it for the final batch only
            return
        no_plotted = 0
        try:
            x, y, names, all_ingredients = batch
        except ValueError:
            x, y, names, all_ingredients = batch[0]
        
        all_data = []
        for prediction, ground_truth, image, name, ingredients in zip(
            outputs[0], y, x, names, all_ingredients):
            bool_prediction = bool(prediction > 0)
            if no_plotted >= self.no_to_plot or bool_prediction == ground_truth:
                continue
            wandb_image = wandb.Image(image)
            this_data = [bool_prediction, bool(ground_truth), float(prediction),
                            wandb_image, name, ingredients]
            all_data.append(this_data)
            no_plotted += 1
        columns = ["Prediction", "Ground truth", "Score", "Image", "Name",
                    "Ingredients"]
        self.wandb_logger.log_table(
            key="Incorrect samples",
            columns=columns,
            data=all_data)