import os
from typing import Dict, Any

import pytorch_lightning as pl
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
from utils.losses import DiceBCELoss

class Experiment(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Experiment, self).__init__()
        # Experiment configuration
        self.cfg = cfg
        # Define subsets
        self.subsets = ['train', 'validation', 'test']
        self.fit_subsets = ['train', 'validation']
        # Define models and loss
        self.model = instantiate(cfg.model.module)
        self.threshold = 0.3
        #self.loss_criterion = F.binary_cross_entropy_with_logits
        self.loss_criterion = DiceBCELoss()

        # Number of models parameters
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.cont_train=0

    def forward(self, batch):
        input, target = batch
        logits = self.model(input)
        indices = torch.where(torch.exp(logits) > self.threshold, 1., 0.) 
        return {"logits": logits, "indices": indices}

    def training_step(self, batch, idx):
        input, target = batch
        out = self.forward(batch)
        loss = self.loss_criterion(out["logits"], target.float())
        return {"loss": loss, "logits": out["logits"], "indices": out["indices"]}

    def validation_step(self, batch, idx, dataloader_idx=0):
        input, target = batch
        target = torch.clamp(target, 0)
        out = self.forward(batch)
        loss = self.loss_criterion(out["logits"], target.float())
        return {"loss": loss, "logits": out["logits"], "indices": out["indices"]}

    def test_step(self, batch, idx, dataloader_idx=0):
        out = self.forward(batch)
        return {"logits": out["logits"], "indices": out["indices"]}

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.train.optimizer,params=self.parameters())
        return {'optimizer': optimizer}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['cfg'] = self.cfg
        checkpoint['current_epoch'] = self.current_epoch
