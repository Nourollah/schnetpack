# Block 1

import os
import typing

import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn
from schnetpack import properties as structure

import torch
import torchmetrics
import pytorch_lightning as pl
# from pytorch_lightning import loggers as pl_loggers
import wandb
from rich.pretty import pprint

# Initialize wandb
wandb.init(project="MLP", entity="KRG-Cardiff")

qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
	os.makedirs(qm9tut)

# Block 2

qm9data = QM9(
	'./qm9.db',
	batch_size=100,
	num_train=20000,
	num_val=1001,
	transforms=[
		trn.ASENeighborList(cutoff=5.),
		trn.RemoveOffsets(QM9.G, remove_mean=True, remove_atomrefs=True),
		trn.CastTo32()
	],
	train_transforms=[
		trn.KNNRepresentation(k=3, threshold=2.0),
	],
	val_transforms=[
		trn.KNNRepresentation(k=3, threshold=2.0),
	],
	test_transforms=[
		trn.KNNRepresentation(k=3, threshold=2.0),
	],
	property_units={
		QM9.G: 'Ha',
	},
	num_workers=2,
	split_file=os.path.join(qm9tut, "split.npz"),
	pin_memory=True,  # set to false, when not using a GPU
	remove_uncharacterized=True,
	load_properties=[
		QM9.G,
	],  # only load U0 property
)
qm9data.prepare_data()
qm9data.setup()


# Defining a model
class MLP(torch.nn.Module):
	def __init__(self,
	             input_size: int,
	             hidden_sizes: typing.List[int],
	             output_size: int):
		super().__init__()
		self._model_dict = torch.nn.ModuleDict()
		self._model_dict["input"] = torch.nn.Linear(input_size, hidden_sizes[0])
		for idx, hid_size in enumerate(hidden_sizes):
			self._model_dict[f"hidden_{idx}"] = torch.nn.Linear(hid_size, hid_size)
			self._model_dict[f"activation_{idx}"] = torch.nn.ReLU(inplace=True)
		self._model_dict["output"] = torch.nn.Linear(hidden_sizes[0], output_size)

	def forward(self, x) -> torch.Tensor:
		for layer in self._model_dict.values():
			x = layer(x)
		return x


mlp_model = MLP(7, [100, 100, 100, 100], 1)
criterion = torch.nn.MSELoss()
optimizer: torch.optim.Adam = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
epochs: int = 10

# print(mlp_model)
for epoch in range(epochs):
	losses: typing.List[int] = []
	for batch in qm9data.train_dataloader():
		# print(batch)
		intermediate: torch.Tensor = torch.stack(
			[mlp_model(j.float()) for j in torch.unbind(batch[structure.knn], dim=1)], dim=1)
		split_intermediate: typing.Tuple[torch.Tensor] = torch.split(intermediate, batch[structure.knn_index], dim=0)
		# @TODO: I believe average is not works for model to learn the situation so I used sum
		summed_split = torch.stack([part.sum() for part in split_intermediate])

		p = summed_split.float()
		t = batch[QM9.G].float()
		loss: torch.Tensor = criterion(p, t)
		losses.append(loss.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Log loss for each batch
		wandb.log({"batch_loss": loss.item()})

	avg_loss = torch.tensor(losses).sum() / len(losses)
	pprint(f"Average loss: {avg_loss}")

	# Log average loss for each epoch
	wandb.log({"epoch_loss": avg_loss})

# Finish wandb run
wandb.finish()
