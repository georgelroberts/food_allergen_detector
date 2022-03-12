import torch
import torch.nn as nn
import sklearn
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class AllergenClassifier(pl.LightningModule):

    def __init__(self):

        super(AllergenClassifier, self).__init__()

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, (5, 5), padding=1),
                nn.MaxPool2d(5),
                nn.ReLU(),
                nn.Conv2d(64, 256, (3, 3), padding=1),
                nn.MaxPool2d(5),
                nn.ReLU(),
                nn.Flatten())

        self.head = nn.Sequential(
                nn.Linear(2048, 32),
                nn.Linear(32, 1),
                nn.Sigmoid())

    def forward(self, x):

        x = self.feature_extractor(x)
        output = self.head(x)

        return output

    def training_step(self, batch, batch_index):

        x, y = batch

        output = self.forward(x)

        loss = nn.BCELoss()(output, y.unsqueeze(1))

        y_pred = output.argmax(-1).cpu().numpy()
        y_tgt = y.cpu().numpy()
        accuracy = accuracy_score(y_tgt, y_pred)
        self.log("train loss", loss)
        self.log("train accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch

        output = self.forward(x)

        loss = nn.BCELoss()(output, y.unsqueeze(1))

        pred = output.argmax(-1)

        return output, pred, y

    def validation_epoch_end(self, validation_step_outputs):

        losses = 0
        outputs = None
        preds = None
        tgts = None
        for output, pred, tgt in validation_step_outputs:
            preds = torch.cat([preds, pred]) if preds is not None else pred
            outputs = torch.cat([outputs, output], dim = 0) \
            if outputs is not None else output
            tgts = torch.cat([tgts, tgt]) if tgts is not None else tgt

        loss = nn.BCELoss()(outputs, tgts.unsqueeze(1))

        y_preds = preds.cpu().numpy()
        y_tgts = tgts.cpu().numpy()


        # pytorch lightning prints a deprecation warning for FM.accuracy,
        # so we'll include sklearn.metrics.accuracy_score as an alternative
        accuracy = accuracy_score(y_tgts, y_preds)

        self.log("val_accuracy", accuracy)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
