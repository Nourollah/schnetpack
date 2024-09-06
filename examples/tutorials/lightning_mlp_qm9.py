import os
import pathlib
import random
import typing

import numpy as np
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn
from schnetpack import properties as structure

import torch
import pytorch_lightning as pl
from torchmetrics import regression
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from rich.pretty import pprint


class QMLP(pl.LightningModule):
	def __init__(self,
	             input_size: int,
	             hidden_sizes: typing.List[int],
	             output_size: int,
	             max_z: int = 101,
	             n_atom_basis: int = 128,
	             learning_rate: float = 0.001):
		super().__init__()
		self.save_hyperparameters()

		# Define the model architecture
		self._model_dict = torch.nn.ModuleDict()
		self._model_dict["input"] = torch.nn.Linear(input_size, 3000)
		self._model_dict["activation_0"] = torch.nn.ReLU(inplace=True)
		self._model_dict["hidden_0"] = torch.nn.Linear(3000, 1000)
		self._model_dict["activation_1"] = torch.nn.ReLU(inplace=True)
		self._model_dict["output"] = torch.nn.Linear(1000, output_size)

		# Embedding layer for atomic numbers
		self._embedding = torch.nn.Embedding(max_z, n_atom_basis, padding_idx=0)
		self.criterion = torch.nn.MSELoss()
		self.atom_rep: int = n_atom_basis
		self.validation_step_losses: typing.List[torch.Tensor] = []

		self.best_train_mae = float('inf')
		self.best_val_mae = float('inf')
		self.mae_table = wandb.Table(columns=["Epoch", "Best Train MAE", "Best Val MAE"])

	def _embedding_step(self, batch: dict) -> dict:
		"""
		Add atom embeddings to the batch
		:param batch: The input batch which contains atomic numbers in knn block
		:return: Inplace is true for this function however the batch is returned for consistency
		"""
		# @TODO: parametrize the hard coded integers in this block
		z_0: torch.Tensor = self._embedding(batch[structure.knn][:, :, 1].type(torch.int))[:, None]
		z_1: torch.Tensor = self._embedding(batch[structure.knn][:, :, 2].type(torch.int))[:, None]
		batch[structure.knn][:, :, 1: self.atom_rep + 1] = torch.squeeze(z_0)
		batch[structure.knn][:, :, self.atom_rep + 1: self.atom_rep * 2 + 1] = torch.squeeze(z_1)
		return batch

	def forward(self, x) -> torch.Tensor:
		for layer in self._model_dict.values():
			x = layer(x)
		return x

	def training_step(self, batch: typing.Dict[str, torch.Tensor], batch_idx: int) -> typing.Dict[str, torch.Tensor]:
		res = self._shared_forward_step(batch)
		p = res.float()
		t = batch[QM9.G].float()
		loss: torch.Tensor = self.criterion(p, t)

		mae = self._calculate_mae(p, t)
		# Update best train MAE
		if mae < self.best_train_mae:
			self.best_train_mae = mae

		self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch: typing.Dict[str, torch.Tensor], batch_idx: int) -> typing.Dict[str, torch.Tensor]:
		res = self._shared_forward_step(batch)

		p = res.float()
		t = batch[QM9.G].float()
		loss: torch.Tensor = self.criterion(p, t)

		mae = self._calculate_mae(p, t)
		# Update best val MAE
		if mae < self.best_val_mae:
			self.best_val_mae = mae

		self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
		return {'val_loss': loss, 'val_mae': mae}

	def test_step(self, batch: typing.Dict[str, torch.Tensor], batch_idx: int) -> typing.Dict[str, torch.Tensor]:
		res = self._shared_forward_step(batch)

		p = res.float()
		t = batch[QM9.G].float()
		loss: torch.Tensor = self.criterion(p, t)

		mae = self._calculate_mae(p, t)

		self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
		self.log('test_mae', mae, on_step=True, on_epoch=True, prog_bar=True)
		return {'test_loss': loss, 'test_mae': mae}

	def on_epoch_end(self):
		self.mae_table.add_data(self.current_epoch, self.best_train_mae.item(), self.best_val_mae.item())
		wandb.log({"Best MAE Table": self.mae_table})

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

	def _shared_forward_step(self, batch: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
		self._embedding_step(batch)
		intermediate: torch.Tensor = torch.stack(
			[self(j.float()) for j in torch.unbind(batch[structure.knn], dim=1)], dim=1)
		split_intermediate: typing.Tuple[torch.Tensor] = torch.split(intermediate, batch[structure.knn_index], dim=0)
		return torch.stack([part.sum() for part in split_intermediate])

	def predict_step(self, batch: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
		res = self._shared_forward_step(batch)
		predictions = res.float()
		return predictions

	def _calculate_mae(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		# Calculate MAE
		mae = torch.nn.functional.l1_loss(predictions, targets)
		return mae


# def on_validation_epoch_end(self) -> None:
# 	avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_losses]).mean()
# 	self.log('val_loss', avg_loss, prog_bar=True)
# 	self.validation_step_losses.clear()


def prepare_data(data_path: pathlib.Path = pathlib.Path('./qm9tut'),
                 k: int = 10,
                 distance_threshold: float = 100.0,
                 num_workers: int = 2,
                 batch_size: int = 1000,
                 num_train: int = 110831,
                 num_val: int = 10000,
                 num_test: int = 10000,
                 seed: int = 42):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	if not os.path.exists(data_path):
		os.makedirs(data_path)

	qm9data = QM9(
		'./qm9.db',
		batch_size=batch_size,
		num_train=num_train,
		num_val=num_val,
		num_test=num_test,
		transforms=[
			trn.ASENeighborList(cutoff=5.),
			trn.RemoveOffsets(QM9.G, remove_mean=True, remove_atomrefs=True),
			trn.CastTo32()
		],
		train_transforms=[trn.KNNRepresentation(k=k, threshold=distance_threshold)],
		val_transforms=[trn.KNNRepresentation(k=k, threshold=distance_threshold)],
		test_transforms=[trn.KNNRepresentation(k=k, threshold=distance_threshold)],
		property_units={QM9.G: 'Ha'},
		num_workers=num_workers,
		split_file=os.path.join(data_path, "split.npz"),
		pin_memory=True,
		remove_uncharacterized=True,
		load_properties=[QM9.G],
	)
	qm9data.prepare_data()
	qm9data.setup()
	return qm9data


def main():
	config = {
		"num_workers": 2,
		"distance_threshold": 100.0,
		"k": 10,
		"input_size": 65,
		"hidden_sizes": [100, 100, 100, 100],
		"output_size": 1,
		"epochs": 100,
		"batch_size": 1000,
		"num_train": 110831,
		"num_val": 10000,
		"num_test": 10000,
	}
	wandb.init(project="MLP",
	           entity="KRG-Cardiff",
	           config=config)

	qm9data = prepare_data(k=config["k"],
	                       distance_threshold=config["distance_threshold"],
	                       num_workers=config["num_workers"],
	                       batch_size=config["batch_size"],
	                       num_train=config["num_train"],
	                       num_val=config["num_val"],
	                       num_test=config["num_test"])

	model = QMLP(input_size=config["input_size"],
	             hidden_sizes=config["hidden_sizes"],
	             output_size=config["output_size"])

	wandb_logger = WandbLogger(project="MLP")

	checkpoint_callback = ModelCheckpoint(
		dirpath='checkpoints',
		filename='mlp-{epoch:02d}-{val_loss:.2f}',
		save_top_k=3,
		monitor='val_loss'
	)

	early_stop_callback = EarlyStopping(
		monitor='val_loss',
		min_delta=0.001,
		patience=3,
		verbose=False,
		mode='min'
	)

	trainer = pl.Trainer(
		max_epochs=config["epochs"],
		logger=wandb_logger,
		callbacks=[checkpoint_callback, early_stop_callback],
		accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
		devices=1,  # Number of GPUs to use
	)

	trainer.fit(model, qm9data.train_dataloader(), qm9data.val_dataloader())
	trainer.test(model, qm9data.test_dataloader())


wandb.finish()

if __name__ == "__main__":
	main()
