import os
import pathlib
import typing
from typing import Dict, Any

import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn
from schnetpack import properties as structure

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from rich.pretty import pprint


class QMLP(pl.LightningModule):
	def __init__(self,
	             input_size: int,
	             hidden_sizes: typing.List[int],
	             output_size: int,
	             max_z: int = 101,
	             n_atom_basis: int = 32,
	             learning_rate: float = 0.001):
		super().__init__()
		self.save_hyperparameters()
		self._model_dict = torch.nn.ModuleDict()
		self._model_dict["input"] = torch.nn.Linear(input_size, hidden_sizes[0])
		for idx, hid_size in enumerate(hidden_sizes):
			self._model_dict[f"hidden_{idx}"] = torch.nn.Linear(hid_size, hid_size)
			self._model_dict[f"activation_{idx}"] = torch.nn.ReLU(inplace=True)
		self._model_dict["output"] = torch.nn.Linear(hidden_sizes[0], output_size)
		self._embedding = torch.nn.Embedding(max_z, n_atom_basis, padding_idx=0)
		self.criterion = torch.nn.MSELoss()
		self.atom_rep: int = n_atom_basis
		self.validation_step_losses: typing.List[torch.Tensor] = []

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

	def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
		self._embedding_step(batch)
		intermediate: torch.Tensor = torch.stack(
			[self(j.float()) for j in torch.unbind(batch[structure.knn], dim=1)], dim=1)
		split_intermediate: typing.Tuple[torch.Tensor] = torch.split(intermediate, batch[structure.knn_index], dim=0)
		summed_split = torch.stack([part.sum() for part in split_intermediate])

		p = summed_split.float()
		t = batch[QM9.G].float()
		loss: torch.Tensor = self.criterion(p, t)

		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

	def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
		self._embedding_step(batch)
		intermediate: torch.Tensor = torch.stack(
			[self(j.float()) for j in torch.unbind(batch[structure.knn], dim=1)], dim=1)
		split_intermediate: typing.Tuple[torch.Tensor] = torch.split(intermediate, batch[structure.knn_index], dim=0)
		summed_split = torch.stack([part.sum() for part in split_intermediate])

		p = summed_split.float()
		t = batch[QM9.G].float()
		loss: torch.Tensor = self.criterion(p, t)

		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
		return {'val_loss': loss}

	# def on_validation_epoch_end(self) -> None:
	# 	avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_losses]).mean()
	# 	self.log('val_loss', avg_loss, prog_bar=True)
	# 	self.validation_step_losses.clear()


def prepare_data(data_path: pathlib.Path = pathlib.Path('./qm9tut')):
	if not os.path.exists(data_path):
		os.makedirs(data_path)

	qm9data = QM9(
		'./qm9.db',
		batch_size=100,
		num_train=50000,
		num_val=10000,
		transforms=[
			trn.ASENeighborList(cutoff=5.),
			trn.RemoveOffsets(QM9.G, remove_mean=True, remove_atomrefs=True),
			trn.CastTo32()
		],
		train_transforms=[trn.KNNRepresentation(k=4, threshold=2.0)],
		val_transforms=[trn.KNNRepresentation(k=4, threshold=2.0)],
		test_transforms=[trn.KNNRepresentation(k=4, threshold=2.0)],
		property_units={QM9.G: 'Ha'},
		num_workers=2,
		split_file=os.path.join(data_path, "split.npz"),
		pin_memory=True,
		remove_uncharacterized=True,
		load_properties=[QM9.G],
	)
	qm9data.prepare_data()
	qm9data.setup()
	return qm9data


def main():
	wandb.init(project="MLP", entity="KRG-Cardiff")

	qm9data = prepare_data()

	model = QMLP(input_size=65, hidden_sizes=[100, 100, 100, 100], output_size=1)

	wandb_logger = WandbLogger(project="MLP")

	checkpoint_callback = ModelCheckpoint(
		dirpath='checkpoints',
		filename='mlp-{epoch:02d}-{val_loss:.2f}',
		save_top_k=3,
		monitor='val_loss'
	)

	trainer = pl.Trainer(
		max_epochs=100,
		logger=wandb_logger,
		callbacks=[checkpoint_callback],
		# accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
		# devices=1,  # Number of GPUs to use
	)

	trainer.fit(model, qm9data.train_dataloader(), qm9data.val_dataloader())


wandb.finish()

if __name__ == "__main__":
	main()
